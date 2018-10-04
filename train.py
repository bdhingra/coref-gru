import numpy as np
import time
import os
import cPickle as pkl
import tensorflow as tf

from model import GAMage
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

    dp = DataPreprocessor.DataPreprocessor()
    data = dp.preprocess_rc(params, dataset)

    print("building minibatch loaders ...")
    batch_loader_train = MiniBatchLoader.MiniBatchLoaderMention(
        params, data.training, params['batch_size'])
    batch_loader_val = MiniBatchLoader.MiniBatchLoaderMention(
        params, data.validation, params['batch_size'], shuffle=False,
        ensure_answer=False)
    batch_loader_test = MiniBatchLoader.MiniBatchLoaderMention(
        params, data.test, params['batch_size'], shuffle=False,
        ensure_answer=False)

    print("building network ...")
    W_init, embed_dim, = Helpers.load_word2vec_embeddings(data.dictionary[0], word2vec)
    m = GAMage.Model(params, W_init, embed_dim)

    print("training ...")
    num_iter = 0
    max_acc = 0.
    min_loss = 1e5

    logger = open(save_path+'/log','a',0)
    train_writer = tf.summary.FileWriter(os.path.join(save_path, 'train'))
    val_writer = tf.summary.FileWriter(os.path.join(save_path, 'val'))

    if params['reload_']:
        print('loading previously saved model')
        saves = pkl.load(open('%s/checkpoints.p'%save_path))
        m.load_model('%s/best_model.p'%save_path, saves[-1])

    # train
    if mode=='train':
        saves = []
        for epoch in xrange(params['num_epochs']):
            estart = time.time()
            stop_flag = False

            for example in batch_loader_train:
                loss, tr_acc, probs, summary = m.train(*example[:-2])

                if num_iter % params['logging_frequency'] == 0:
                    message = ("Epoch %d TRAIN loss=%.4e acc=%.4f elapsed=%.1f" % (
                        epoch, loss, tr_acc, time.time()-estart))
                    print message
                    logger.write(message+'\n')
                    train_writer.add_summary(summary, num_iter)

                num_iter += 1
                if num_iter % params['validation_frequency'] == 0:
                    total_loss, total_acc, n = 0., 0., 0.

                    for example in batch_loader_val:
                        outs = m.validate(*example[:-2])
                        loss, acc, probs = outs[:3]

                        bsize = example[0].shape[0]
                        total_loss += bsize*loss
                        total_acc += bsize*acc
                        n += bsize

                    val_acc = total_acc/n
                    if val_acc > max_acc:
                        max_acc = val_acc
                        save_id = num_iter
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

                if num_iter % params["anneal_frequency"] == 0:
                    m.anneal()

            #m.save_model('%s/model_%d.p'%(save_path,epoch))
            message = "After Epoch %d: Train acc=%.4f, Val acc=%.4f" % (epoch, tr_acc, max_acc)
            print message
            logger.write(message+'\n')
            
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
    print('loading model')
    m.load_model('%s/best_model.p'%save_path, saves[-1])

    total_loss, total_acc, n = 0., 0., 0
    answer_structure = {}
    idict = data.inv_dictionary
    for example in batch_loader_val:
        outs = m.validate(*example[:-2])
        loss, acc, probs = outs[:3]

        pred_indices = np.argmax(probs, axis=1)
        for i in range(len(example[-1])):
            cname = str(example[-1][i]).strip()
            gt_answer = example[10][i]
            answer_structure[cname] = (pred_indices[i], gt_answer, probs[i, :])

        bsize = example[0].shape[0]
        total_loss += bsize*loss
        total_acc += bsize*acc

        n += bsize
    test_acc = total_acc/n
    test_loss = total_loss/n
    message = "TEST loss=%.4e acc=%.4f" % (test_loss, test_acc)
    print message
    logger.write(message+'\n')
    pkl.dump(answer_structure,
            open(os.path.join(save_path, "test_answer_structure.p"), "w"))

    logger.close()

    # clean up
    print("Cleaning up saved models ...")
