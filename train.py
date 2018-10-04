import numpy as np
import time
import os
import cPickle as pkl
import tensorflow as tf

from model import GA
from utils import Helpers, DataPreprocessor, MiniBatchLoader

def _add_summary(writer, scalar, tag, step):
    """Add summary to writer for given value and tag.

    Args:
        writer: TensorFlow summary FileWriter object.
        scalar: Scalar value to log.
        tag: Label for the value.
        step: Integer step for logging.
    """
    summary = tf.Summary()
    value = summary.value.add()
    value.simple_value = scalar
    value.tag = tag
    writer.add_summary(summary, step)
    writer.flush()

def main(save_path, params, mode='train'):

    word2vec = params['word2vec']
    dataset = params['dataset']

    use_chars = params['char_dim']>0
    dp = DataPreprocessor.DataPreprocessor()
    nts = True if mode in ['test','validation'] else False
    data = dp.preprocess(dataset, max_chains=params['max_chains'], 
                         no_training_set=nts, use_chars=use_chars)
    cloze = params['cloze']

    print("building minibatch loaders ...")
    if mode=='train':
        batch_loader_train = MiniBatchLoader.MiniBatchLoader(
            data.training, params['batch_size'], 
            data.max_num_cand, params['max_chains'])
        batch_loader_val = MiniBatchLoader.MiniBatchLoader(
            data.validation, params['batch_size'], 
            data.max_num_cand, params['max_chains'])
        batch_loader_test = MiniBatchLoader.MiniBatchLoader(
            data.test, params['batch_size'], 
            data.max_num_cand, params['max_chains'], shuffle=False)
    if mode=='test':
        batch_loader_test = MiniBatchLoader.MiniBatchLoader(
            data.test, params['batch_size'], 
            data.max_num_cand, params['max_chains'], shuffle=False)
    if mode=='validation':
        batch_loader_test = MiniBatchLoader.MiniBatchLoader(
            data.validation, params['batch_size'], 
            data.max_num_cand, params['max_chains'], shuffle=False)
    num_candidates = data.max_num_cand

    print("building network ...")
    W_init, embed_dim, = Helpers.load_word2vec_embeddings(data.dictionary[0], word2vec)
    m = GA.Model(params, data.vocab_size, data.num_chars, W_init, embed_dim,
                 num_candidates, cloze=cloze)

    print("training ...")
    num_iter = 0
    max_acc = 0.
    min_loss = 1e5
    deltas = []

    logger = open(save_path+'/log','a',0)
    train_writer = tf.summary.FileWriter(os.path.join(save_path, 'train'))
    val_writer = tf.summary.FileWriter(os.path.join(save_path, 'val'))

    if params['reload_'] and mode=='train':
        print('loading previously saved model')
        saves = pkl.load(open('%s/checkpoints.p'%save_path))
        m.load_model('%s/best_model.p'%save_path, saves[-1])

    # train
    if mode=='train':
        tafter = 0.
        saves = []
        for epoch in xrange(params['num_epochs']):
            estart = time.time()
            new_max = False
            stop_flag = False

            for (dw, dt, qw, qt, a, m_dw, m_qw, tt, tm, c, m_c, 
                    cl, crd, crq, fnames) in batch_loader_train:
                tc = time.time()-tafter
                loss, tr_acc, probs, summary = m.train(dw, dt, qw, qt, c, a, m_dw, 
                        m_qw, tt, tm, m_c, cl, crd, crq)
                tafter = time.time()

                if num_iter % params['logging_frequency'] == 0:
                    message = ("Epoch %d TRAIN loss=%.4e acc=%.4f elapsed=%.1f"
                               " (%.1f outside)" % (epoch, loss, tr_acc,
                                         time.time()-estart, tc))
                    print message
                    logger.write(message+'\n')
                    train_writer.add_summary(summary, num_iter)

                num_iter += 1
                if num_iter % params['validation_frequency'] == 0:
                    total_loss, total_acc, n, n_cand = 0., 0., 0., 0.

                    for (dw, dt, qw, qt, a, m_dw, m_qw, tt, tm, c, m_c, 
                            cl, crd, crq, fnames) in batch_loader_val:
                        outs = m.validate(dw, dt, qw, qt, c, a, 
                                m_dw, m_qw, tt, tm, m_c, cl, crd, crq)
                        loss, acc, probs = outs[:3]

                        bsize = dw.shape[0]
                        total_loss += bsize*loss
                        total_acc += bsize*acc
                        n += bsize

                    val_acc = total_acc/n
                    if val_acc > max_acc:
                        max_acc = val_acc
                        save_id = num_iter if epoch>0 else 0
                        sv = m.save_model('%s/best_model.p'%save_path, save_id)
                        saves.append(save_id)
                        new_max = True
                    val_loss = total_loss/n
                    message = "Epoch %d VAL loss=%.4e acc=%.4f max_acc=%.4f" % (
                        epoch, val_loss, val_acc, max_acc)
                    print message
                    logger.write(message+'\n')

                    _add_summary(val_writer, val_loss, "loss", num_iter)
                    _add_summary(val_writer, val_acc, "accuracy", num_iter)

                    # stopping
                    if val_loss<min_loss: min_loss = val_loss
                    if params['stopping_criterion'] and (
                            val_loss-min_loss)/min_loss>0.3:
                        stop_flag = True
                        break

            message = "After Epoch %d: Train acc=%.4f, Val acc=%.4f" % (epoch, tr_acc, max_acc)
            print message
            logger.write(message+'\n')
            
            # learning schedule
            if epoch >=2 and epoch % params['anneal_frequency'] == 0:
                m.anneal()
            if stop_flag: break
        # record all saved models
        pkl.dump(saves, open('%s/checkpoints.p'%save_path,'w'))

    # test
    mode = 'test' if mode in ['train','test'] else 'val'
    print("testing ...")
    try:
        saves = pkl.load(open('%s/checkpoints.p'%save_path))
    except IOError:
        def _to_num(foo):
            try: num = int(foo)
            except ValueError: return None
            return num

        saves = []
        for directory in os.listdir(save_path):
            if not os.path.isdir(os.path.join(save_path, directory)): continue
            num = _to_num(directory)
            if num is None: continue
            saves.append(num)

        saves = sorted(saves)
    if not saves:
        print("No models saved during training!")
        return

    m.load_model('%s/best_model.p'%save_path, saves[-1])

    fids = []
    total_loss, total_acc, n = 0., 0., 0
    answer = np.zeros((len(batch_loader_test.questions),)).astype('float32')
    preds = np.zeros((len(batch_loader_test.questions),)).astype('float32')
    for dw, dt, qw, qt, a, m_dw, m_qw, tt, tm, c, m_c, cl, crd, crq, fnames in batch_loader_test:
        outs = m.validate(dw, dt, qw, qt, c, a, m_dw, m_qw, tt, tm, m_c, cl, crd, crq)
        loss, acc, probs, drep, qrep, doc_probs = outs[:6]

        bsize = dw.shape[0]
        total_loss += bsize*loss
        total_acc += bsize*acc

        answer[n:n+bsize] = a
        preds[n:n+bsize] = np.argmax(probs,axis=1)
        fids += fnames
        n += bsize
    message = '%s Loss %.4e Acc %.4f' % (mode.upper(), total_loss/n, total_acc/n)
    print message
    logger.write(message+'\n')

    np.save('%s/%s.answer' % (save_path,mode), answer)
    np.save('%s/%s.preds' % (save_path,mode), preds)
    f = open('%s/%s.ids' % (save_path,mode),'w')
    for item in fids: f.write(str(item)+'\n')
    f.close()
    logger.close()
