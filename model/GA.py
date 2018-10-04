import os
import tensorflow as tf
import numpy as np
from tflayers import *

EPS = 1e-7

def prepare_input(d,q):
    f = np.zeros(d.shape).astype('int32')
    for i in range(d.shape[0]):
        f[i,:] = np.in1d(d[i,:],q[i,:])
    return f

class Model(object):

    def __init__(self, params, W_init, embed_dim, device=0):
        self.use_edgetypes = True
        self.embed_dim = embed_dim
        self.dropout = params['dropout']
        self.train_emb = params['train_emb']
        self.char_dim = params['char_dim']
        self.learning_rate = params['learning_rate']
        self.num_chars = params["num_characters"]
        self.char_filter_size = params["char_filter_size"]
        self.char_filter_width = params["char_filter_width"]
        self.use_feat = params['use_feat']
        self.max_chains = params['max_chains']
        self.max_word_len = params['max_word_len']
        self.nhidden = params['num_relations'] * params['relation_dims']
        if self.use_edgetypes:
            self.num_relations = params['num_relations']
            self.relation_dims = params['relation_dims']
        else:
            self.num_relations = 1
            self.relation_dims = self.nhidden
        seed = params['seed']
        K = params['nlayers']

        self.graph = tf.Graph()
        with self.graph.as_default(), tf.device("/gpu:%d" % device):
            tf.set_random_seed(seed)
            # placeholders
            self.doc = tf.placeholder(tf.int32, shape=(None, None))
            self.doc_chars = tf.placeholder(tf.int32, shape=(None, None, self.max_word_len))
            self.docei = tf.placeholder(tf.float32, shape=(None, None, self.max_chains))
            self.doceo = tf.placeholder(tf.float32, shape=(None, None, self.max_chains))
            self.docri = tf.placeholder(tf.int32, shape=(None, None, self.max_chains))
            self.docro = tf.placeholder(tf.int32, shape=(None, None, self.max_chains))
            self.qry = tf.placeholder(tf.int32, shape=(None, None))
            self.qry_chars = tf.placeholder(tf.int32, shape=(None, None, self.max_word_len))
            self.cand = tf.placeholder(tf.int32, shape=(None, None, None))
            self.dmask = tf.placeholder(tf.float32, shape=(None, None))
            self.qmask = tf.placeholder(tf.float32, shape=(None, None))
            self.cmask = tf.placeholder(tf.float32, shape=(None, None))
            self.ans = tf.placeholder(tf.int32, shape=(None))
            self.feat = tf.placeholder(tf.int32, shape=(None, None))
            self.keep_prob = tf.placeholder(tf.float32)
            self.lrate = tf.placeholder(tf.float32)
            
            # variables
            if W_init is None:
                W_init = tf.random_normal((params["vocab_size"], self.embed_dim), 
                        mean=0.0, stddev=glorot(params["vocab_size"], self.embed_dim), 
                        dtype=tf.float32)
            self.Wemb = tf.Variable(W_init, trainable=bool(self.train_emb))
            self.Femb = tf.Variable(tf.random_normal((2,2), mean=0.0, stddev=glorot(2,2), 
                dtype=tf.float32))

            # network
            # embeddings
            doc_emb = tf.nn.embedding_lookup(self.Wemb, self.doc) # B x N x De
            doc_char_emb = self.get_character_embeddings(self.doc_chars)
            doc_emb = tf.concat([doc_emb, doc_char_emb], axis=2)
            qry_emb = tf.nn.embedding_lookup(self.Wemb, self.qry) # B x Q x De
            qry_char_emb = self.get_character_embeddings(self.qry_chars, reuse=True)
            qry_emb = tf.concat([qry_emb, qry_char_emb], axis=2)
            fea_emb = tf.nn.embedding_lookup(self.Femb, self.feat) # B x N x 2
            self.aggs = []
            # layers
            for i in range(K):
                # append feat
                indoc = self.embed_dim + self.char_filter_size if i==0 else 2*self.nhidden
                if self.use_feat and i==K-1:
                    doc_emb = tf.concat([doc_emb, fea_emb], axis=2) # B x N x (De+2)
                    indoc += 2
                # forward
                fdoc = CorefGRU(self.num_relations, indoc, self.relation_dims, 
                        self.max_chains)
                fdout, dmem, fdagg = fdoc.compute(doc_emb, self.dmask, self.docei, self.doceo, 
                        self.docri, self.docro) # B x N x Dh
                # backward
                # flip masks o<->i, mirror relation types
                bdoc = CorefGRU(self.num_relations, indoc, self.relation_dims, 
                        self.max_chains, reverse=True)
                bdout, dmem, bdagg = bdoc.compute(doc_emb, self.dmask, self.doceo, self.docei, 
                        self.docro, self.docri) # B x N x Dh
                doc_emb = tf.concat([fdout, bdout], axis=2) # B x N x 2Dh
                # qry
                inqry = self.embed_dim + self.char_filter_size if i==0 else 2*self.nhidden
                fgru = GRU(inqry, self.nhidden, "qryfgru%d"%i)
                bgru = GRU(inqry, self.nhidden, "qrybgru%d"%i, reverse=True)
                fout = fgru.compute(None, qry_emb, self.qmask) # B x Q x Dh
                bout = bgru.compute(None, qry_emb, self.qmask) # B x Q x Dh
                qry_emb = tf.concat([fout, bout], axis=2) # B x Q x 2Dh
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
            mid = self.nhidden
            q = tf.concat([qry_emb[:,-1,:mid], qry_emb[:,0,mid:]], axis=1) # B x 2Dh
            p = tf.squeeze(tf.matmul(doc_emb, tf.expand_dims(q,axis=2)),axis=2) # B x N
            probs = tf.nn.softmax(p) # B x N
            probm = probs*self.cmask + EPS
            probm = probm/tf.reduce_sum(probm, axis=1, keep_dims=True) # B x N
            self.probc = tf.squeeze(
                    tf.matmul(tf.expand_dims(probm,axis=1), tf.to_float(self.cand)),axis=1) # B x C

            # loss
            t1hot = tf.one_hot(self.ans, tf.shape(self.probc)[1]) # B x C
            self.correct_probs = tf.reduce_sum(
                tf.to_float(t1hot) * self.probc, axis=1)
            self.instance_loss = tf.reduce_sum(
                tf.to_float(t1hot) * tf.log(self.probc + EPS), axis=1)
            self.loss = - tf.reduce_mean(self.instance_loss)
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
            sess_config = tf.ConfigProto(allow_soft_placement=True)
            sess_config.gpu_options.allow_growth = True
            self.session = tf.Session(config=sess_config)
            self.saver = tf.train.Saver()
            self.session.run(tf.global_variables_initializer())
            self.doc_probs = probs
            self.merged_summaries = tf.summary.merge_all()

    def get_character_embeddings(self, characters, reuse=False):
        """Compute character level representation.

        Args:
            characters: B x N x W Tensor, where W is max_word_len. Assumes
                zero padding of words shorter than length W.
            reuse: If True, all parameters are reused from previously
                initialized values.

        Returns:
            embeddings: B x N x d Tensor.
        """
        with tf.variable_scope("char_embeddings", reuse=reuse):
            char_emb_mat = tf.get_variable(
                "char_emb_mat", shape=[self.num_chars, self.char_dim],
                dtype=tf.float32)
            char_emb = tf.nn.embedding_lookup(char_emb_mat, characters) # B x N x W x dc
            embeddings = self.conv1d(char_emb, self.char_filter_size,
                                     self.char_filter_width, "VALID") # B x N x d
        return embeddings

    @staticmethod
    def conv1d(in_, filter_size, height, padding, scope=None):
        """Code obtained from:
        https://github.com/allenai/bi-att-flow/blob/master/my/tensorflow/nn.py
        """
        with tf.variable_scope(scope or "conv1d"):
            num_channels = in_.get_shape()[-1]
            filter_ = tf.get_variable(
                "filter", shape=[1, height, num_channels, filter_size],
                dtype=tf.float32)
            strides = [1, 1, 1, 1]
            xxc = tf.nn.conv2d(in_, filter_, strides, padding)
            out = tf.reduce_max(tf.nn.relu(xxc), axis=2)  # [-1, JX, d]
        return out

    def anneal(self):
        print "annealing learning rate"
        self.learning_rate /= 2

    @staticmethod
    def remove_edgetypes(ri, ro):
        return np.zeros_like(ri), np.zeros_like(ro)

    @staticmethod
    def get_graph(edges):
        dei, deo = edges
        dri, dro = np.copy(dei).astype("int32"), np.copy(deo).astype("int32")
        dri[:, :, 0] = 0
        dro[:, :, 0] = 0
        return dei, deo, dri, dro

    def train(self, dw, m_dw, qw, m_qw, dc, m_dc, qc, m_qc, cd, m_cd, a, edges):
        f = prepare_input(dw,qw)
        dei, deo, dri, dro = self.get_graph(edges)
        if not self.use_edgetypes:
            # label  all relation types as 0
            dri, dro = self.remove_edgetypes(dri, dro)
        loss, acc, probs, _, summary = self.session.run(
            [self.loss, self.acc, self.probc, self.train_op,
             self.merged_summaries], 
            feed_dict = {
                self.doc : dw,
                self.doc_chars : dc,
                self.docei : dei,
                self.doceo : deo,
                self.docri : dri,
                self.docro : dro,
                self.qry : qw,
                self.qry_chars : qc,
                self.cand : cd,
                self.dmask : m_dw,
                self.qmask : m_qw,
                self.cmask : m_cd,
                self.ans : a,
                self.feat : f,
                self.keep_prob : 1.-self.dropout,
                self.lrate : self.learning_rate,
            })
        return loss, acc, probs, summary

    def validate(self, dw, m_dw, qw, m_qw, dc, m_dc, qc, m_qc, cd, m_cd, a, edges):
        f = prepare_input(dw,qw)
        dei, deo, dri, dro = self.get_graph(edges)
        if not self.use_edgetypes:
            # label  all relation types as 0
            dri, dro = self.remove_edgetypes(dri, dro)
            qri, qro = self.remove_edgetypes(qri, qro)
        outs = self.session.run(
            [self.loss, self.acc, self.probc], 
            feed_dict = {
                self.doc : dw,
                self.doc_chars : dc,
                self.docei : dei,
                self.doceo : deo,
                self.docri : dri,
                self.docro : dro,
                self.qry : qw,
                self.qry_chars : qc,
                self.cand : cd,
                self.dmask : m_dw,
                self.qmask : m_qw,
                self.cmask : m_cd,
                self.ans : a,
                self.feat : f,
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
        if step is not None:
            base_path = load_path.rsplit('/',1)[0]+'/'+str(step)+'/'
        else:
            base_path = load_path.rsplit('/',1)[0]+'/'
        print "loading model from %s ..." %base_path
        new_saver = tf.train.import_meta_graph(base_path+'model.meta')
        new_saver.restore(self.session, tf.train.latest_checkpoint(base_path))
