import os
import tensorflow as tf
import numpy as np
from tflayers import *

EPS = 1e-30

def prepare_input(d,q):
    f = np.zeros(d.shape[:2]).astype('int32')
    for i in range(d.shape[0]):
        f[i,:] = np.in1d(d[i,:,0],q[i,:,0])
    return f

class Model:

    def __init__(self, params, vocab_size, num_chars, W_init, embed_dim, num_cand,
            cloze=True):
        self.use_edgetypes = True
        self.embed_dim = embed_dim
        self.dropout = params['dropout']
        self.train_emb = params['train_emb']
        self.char_dim = params['char_dim']
        self.learning_rate = params['learning_rate']
        self.num_chars = num_chars
        self.use_feat = params['use_feat']
        self.max_chains = params['max_chains']
        self.nhidden = params['num_relations'] * params['relation_dims']
        if self.use_edgetypes:
            self.num_relations = params['num_relations']
            self.relation_dims = params['relation_dims']
        else:
            self.num_relations = 1
            self.relation_dims = self.nhidden
        self.concat = False
        seed = params['seed']
        K = params['nlayers']

        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(seed)
            # placeholders
            self.doc = tf.placeholder(tf.int32, shape=(None, None))
            self.docei = tf.placeholder(tf.float32, shape=(None, None, self.max_chains))
            self.doceo = tf.placeholder(tf.float32, shape=(None, None, self.max_chains))
            self.docri = tf.placeholder(tf.int32, shape=(None, None, self.max_chains))
            self.docro = tf.placeholder(tf.int32, shape=(None, None, self.max_chains))
            self.qry = tf.placeholder(tf.int32, shape=(None, None))
            self.qryei = tf.placeholder(tf.float32, shape=(None, None, self.max_chains))
            self.qryeo = tf.placeholder(tf.float32, shape=(None, None, self.max_chains))
            self.qryri = tf.placeholder(tf.int32, shape=(None, None, self.max_chains))
            self.qryro = tf.placeholder(tf.int32, shape=(None, None, self.max_chains))
            self.cand = tf.placeholder(tf.int32, shape=(None, None, num_cand))
            self.dmask = tf.placeholder(tf.float32, shape=(None, None))
            self.qmask = tf.placeholder(tf.float32, shape=(None, None))
            self.cmask = tf.placeholder(tf.float32, shape=(None, None))
            self.ans = tf.placeholder(tf.int32, shape=(None))
            self.feat = tf.placeholder(tf.int32, shape=(None, None))
            self.cloze = tf.placeholder(tf.int32, shape=(None))
            self.keep_prob = tf.placeholder(tf.float32)
            self.lrate = tf.placeholder(tf.float32)
            
            # variables
            if W_init is None:
                W_init = tf.random_normal((vocab_size, self.embed_dim), 
                        mean=0.0, stddev=glorot(vocab_size, self.embed_dim), 
                        dtype=tf.float32)
            self.Wemb = tf.Variable(W_init, trainable=bool(self.train_emb))
            self.Femb = tf.Variable(tf.random_normal((2,2), mean=0.0, stddev=glorot(2,2), 
                dtype=tf.float32))

            # network
            # embeddings
            doc_emb = tf.nn.embedding_lookup(self.Wemb, self.doc) # B x N x De
            qry_emb = tf.nn.embedding_lookup(self.Wemb, self.qry) # B x Q x De
            fea_emb = tf.nn.embedding_lookup(self.Femb, self.feat) # B x N x 2
            # layers
            for i in range(K):
                # append feat
                indoc = self.embed_dim if i==0 else 2*self.nhidden
                if self.use_feat and i==K-1:
                    doc_emb = tf.concat([doc_emb, fea_emb], axis=2) # B x N x (De+2)
                    indoc += 2
                inqry = self.embed_dim if i==0 else 2*self.nhidden
                # forward
                fdoc = CorefGRU(self.num_relations, indoc, self.relation_dims, 
                        self.max_chains, concat=self.concat)
                fqry = CorefGRU(self.num_relations, inqry, self.relation_dims, 
                        self.max_chains, concat=self.concat)
                fdout, dmem, fdagg = fdoc.compute(doc_emb, self.dmask, self.docei, self.doceo, 
                        self.docri, self.docro) # B x N x Dh
                fqout, qmem, fqagg = fqry.compute(qry_emb, self.qmask, self.qryei, self.qryeo, 
                        self.qryri, self.qryro, mem_init=dmem[:,-1,:,:]) # B x Q x Dh
                # backward
                # flip masks o<->i, mirror relation types
                bdoc = CorefGRU(self.num_relations, indoc, self.relation_dims, 
                        self.max_chains, reverse=True, concat=self.concat)
                bqry = CorefGRU(self.num_relations, inqry, self.relation_dims, 
                        self.max_chains, reverse=True, concat=self.concat)
                bqout, qmem, bqagg = bqry.compute(qry_emb, self.qmask, self.qryeo, self.qryei, 
                        self.qryro, self.qryri)
                bdout, dmem, bdagg = bdoc.compute(doc_emb, self.dmask, self.doceo, self.docei, 
                        self.docro, self.docri, mem_init=qmem[:,-1,:,:]) # B x N x Dh
                doc_emb = tf.concat([fdout, bdout], axis=2) # B x N x 2Dh
                qry_emb = tf.concat([fqout, bqout], axis=2) # B x Q x 2Dh
                # gated attention
                if i<K-1:
                    qshuf = tf.transpose(qry_emb, perm=(0,2,1)) # B x 2Dh x Q
                    M = tf.matmul(doc_emb, qshuf) # B x N x Q
                    alphas = tf.nn.softmax(M)*tf.expand_dims(self.qmask, 
                            axis=1)
                    alphas = alphas/tf.reduce_sum(alphas, 
                            axis=2, keep_dims=True) # B x N x Q
                    gating = tf.matmul(alphas, qry_emb) # B x N x 2Dh
                    doc_emb = doc_emb*gating # B x N x 2Dh
                    doc_emb = tf.nn.dropout(doc_emb, self.keep_prob) 
            # attention sum
            if cloze:
                cl = tf.one_hot(self.cloze, tf.shape(self.qry)[1]) # B x Q
                q = tf.squeeze(tf.matmul(tf.expand_dims(cl,axis=1), qry_emb),axis=1) # B x 2Dh
            else:
                mid = self.nhidden
                q = tf.concat([qry_emb[:,-1,:mid], qry_emb[:,0,mid:]], axis=1) # B x 2Dh
            p = tf.squeeze(tf.matmul(doc_emb, tf.expand_dims(q,axis=2)),axis=2) # B x N
            probs = tf.nn.softmax(p) # B x N
            probm = probs*self.cmask + EPS
            probm = probm/tf.reduce_sum(probm, axis=1, keep_dims=True) # B x N
            self.probc = tf.squeeze(
                    tf.matmul(tf.expand_dims(probm,axis=1), tf.to_float(self.cand)),axis=1) # B x C

            # loss
            t1hot = tf.one_hot(self.ans, num_cand) # B x C
            self.loss = -tf.reduce_mean(tf.reduce_sum(
                tf.to_float(t1hot)*tf.log(self.probc+EPS), axis=1))
            self.acc = tf.reduce_mean(tf.cast(
                tf.equal(tf.cast(tf.argmax(self.probc,axis=1),tf.int32),self.ans), tf.float32))

            # summaries
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('accuracy', self.acc)

            # ops
            opt = tf.train.AdamOptimizer(learning_rate=self.lrate)
            grads = opt.compute_gradients(self.loss)
            grads_clipped = [(tf.clip_by_value(gg, -params['grad_clip'],
                                               params['grad_clip']),var) 
                    for gg,var in grads if gg is not None]
            self.train_op = opt.apply_gradients(grads_clipped)

            # bells and whistles
            self.session = tf.Session()
            self.saver = tf.train.Saver()
            self.session.run(tf.global_variables_initializer())
            self.doc_rep = doc_emb
            self.qry_rep = qry_emb
            self.doc_probs = probs
            self.merged_summaries = tf.summary.merge_all()

    def anneal(self):
        self.learning_rate /= 2

    @staticmethod
    def remove_edgetypes(ri, ro):
        return np.zeros_like(ri), np.zeros_like(ro)

    def train(self, dw, dt, qw, qt, c, a, m_dw, m_qw, tt, tm, m_c, cl, crd, crq):
        f = prepare_input(dw,qw)
        dei, deo, dri, dro = crd
        qei, qeo, qri, qro = crq
        if not self.use_edgetypes:
            # label  all relation types as 0
            dri, dro = self.remove_edgetypes(dri, dro)
            qri, qro = self.remove_edgetypes(qri, qro)
        loss, acc, probs, _, summary = self.session.run(
            [self.loss, self.acc, self.probc, self.train_op,
             self.merged_summaries], 
            feed_dict = {
                self.doc : dw[:,:,0],
                self.docei : dei,
                self.doceo : deo,
                self.docri : dri,
                self.docro : dro,
                self.qry : qw[:,:,0],
                self.qryei : qei,
                self.qryeo : qeo,
                self.qryri : qri,
                self.qryro : qro,
                self.cand : c,
                self.dmask : m_dw.astype('float32'),
                self.qmask : m_qw.astype('float32'),
                self.cmask : m_c.astype('float32'),
                self.ans : a,
                self.feat : f,
                self.cloze : cl,
                self.keep_prob : 1.-self.dropout,
                self.lrate : self.learning_rate,
            })
        return loss, acc, probs, summary

    def validate(self, dw, dt, qw, qt, c, a, m_dw, m_qw, tt, tm, m_c, cl, crd, crq):
        f = prepare_input(dw,qw)
        dei, deo, dri, dro = crd
        qei, qeo, qri, qro = crq
        if not self.use_edgetypes:
            # label  all relation types as 0
            dri, dro = self.remove_edgetypes(dri, dro)
            qri, qro = self.remove_edgetypes(qri, qro)
        outs = self.session.run(
            [self.loss, self.acc, self.probc, self.doc_rep, self.qry_rep,
             self.doc_probs], 
            feed_dict = {
                self.doc : dw[:,:,0],
                self.docei : dei,
                self.doceo : deo,
                self.docri : dri,
                self.docro : dro,
                self.qry : qw[:,:,0],
                self.qryei : qei,
                self.qryeo : qeo,
                self.qryri : qri,
                self.qryro : qro,
                self.cand : c,
                self.dmask : m_dw.astype('float32'),
                self.qmask : m_qw.astype('float32'),
                self.cmask : m_c.astype('float32'),
                self.ans : a,
                self.feat : f,
                self.cloze : cl,
                self.keep_prob : 1.,
            })
        return outs

    def list_of_variables(self):
        with self.graph.as_default():
            return self.session.run(tf.trainable_variables())

    def assign_variables(self, weights):
        with self.graph.as_default():
            assign_ops = []
            for ii,vv in enumerate(tf.trainable_variables()):
                assign_ops.append(vv.assign(weights[ii]))
            self.session.run(assign_ops)
    
    def save_model(self, save_path, step):
        base_path = save_path.rsplit('/',1)[0]+'/'+str(step)+'/'
        if not os.path.exists(base_path): os.makedirs(base_path)
        return self.saver.save(self.session, base_path+'model')

    def load_model(self, load_path, step):
        base_path = load_path.rsplit('/',1)[0]+'/'+str(step)+'/'
        print "loading model from %s ..." %base_path
        new_saver = tf.train.import_meta_graph(base_path+'model.meta')
        new_saver.restore(self.session, tf.train.latest_checkpoint(base_path))
