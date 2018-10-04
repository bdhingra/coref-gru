import tensorflow as tf
import numpy as np

def glorot(d1,d2):
    return np.sqrt(6./(d1+d2))

class GRU(object):
    def __init__(self, idim, odim, layername, reverse=False, reuse=False):
        def _gate_params(insize, outsize, name):
            gate = {}
            gate["W"] = tf.get_variable(
                "W"+name,
                initializer=tf.random_normal(
                    (insize,outsize), 
                    mean=0.0, stddev=glorot(insize,outsize)),
                dtype=tf.float32)
            gate["U"] = tf.get_variable(
                "U"+name,
                initializer=tf.random_normal(
                    (outsize,outsize),
                    mean=0.0, stddev=glorot(outsize,outsize)),
                dtype=tf.float32)
            gate["b"] = tf.get_variable(
                "b"+name,
                initializer=tf.zeros((outsize,)),
                dtype=tf.float32)
            return gate

        with tf.variable_scope(layername, reuse=reuse):
            self.resetgate = _gate_params(idim, odim, "r")
            self.updategate = _gate_params(idim, odim, "u")
            self.hiddengate = _gate_params(idim, odim, "h")
        self.Wstacked = tf.concat([self.resetgate["W"], self.updategate["W"],
                self.hiddengate["W"]], axis=1) # Din x 3Dout
        self.Ustacked = tf.concat([self.resetgate["U"], self.updategate["U"],
                self.hiddengate["U"]], axis=1) # Dr x 3Dout
        self.reverse = reverse
        self.out_dim = odim

    def _gru_cell(self, prev, inp, rgate, ugate, hgate):
        def _slice(a, n):
            s = a[:,n*self.out_dim:(n+1)*self.out_dim]
            return s
        hid_to_hid = tf.matmul(prev, self.Ustacked) # B x 3Dout
        r = tf.sigmoid(_slice(inp,0) + _slice(hid_to_hid,0) + rgate["b"])
        z = tf.sigmoid(_slice(inp,1) + _slice(hid_to_hid,1) + ugate["b"])
        ht = tf.tanh(_slice(inp,2) + r*_slice(hid_to_hid,2) + hgate["b"])
        h = (1.-z)*prev + z*ht
        return h

    def _step_gru(self, prev, inps):
        # prev : B x Do
        # inps : (B x Di, B)
        elems, mask = inps[0], inps[1]
        new = self._gru_cell(prev, elems, 
                self.resetgate, self.updategate, self.hiddengate)
        new.set_shape([None,self.out_dim])
        newmasked = tf.expand_dims((1.-tf.to_float(mask)),axis=1)*prev + \
                tf.expand_dims(tf.to_float(mask),axis=1)*new
        return newmasked

    def compute(self, init, inp, mask):
        # init : B x Do
        # inp : B x N x Di
        # mask : B x N
        if self.reverse:
            inp = tf.reverse(inp, [1])
            mask = tf.reverse(mask, [1])
        if init is None:
            init = tf.zeros((tf.shape(inp)[0],self.out_dim), dtype=tf.float32)
        inpre = tf.transpose(inp, perm=(1,0,2)) # N x B x Di
        maskre = tf.transpose(mask, perm=(1,0)) # N x B
        # precompute input
        Xpre = tf.tensordot(inpre, self.Wstacked, axes=[[2],[0]]) # N x B x 3Dout
        outs = tf.transpose(tf.scan(self._step_gru, (Xpre, maskre), 
                initializer=init), perm=(1,0,2)) # B x N x Do
        if self.reverse:
            outs = tf.reverse(outs, [1])
        return outs

class MageRNN(object):
    """
    MAGE RNN model which takes as input five tensors:
    X: B x N x Din (batch of sequences)
    M: B x N (masks for sequences)
    Ei: B x N x C (one-hot mask for incoming edges at each timestep)
    Eo: B x N x C (one-hot mask for outgoing edges at each timestep)
    Ri: B x N x C (index for incoming relation types at each timestep)
    Ro: B x N x C (index for outgoing relation types at each timestep)
    Q: B x Dout (optionally, query vectors for modulating relation attention)
    C is the maximum number of chains.
    Indexes in Ri/o go from 0 to M-1, where M is the number of relation types

    During each recurrent update, the relevant hidden states are fetched from the memory
    based on R/E, and combined through an attention mechanism. Multiple edges of the same type
    are averaged together.

    During construction provide the parameters for initializing the variables.
        "num_relations" :   total number of relations,
        "input_dim"     :   input dimensionality
        "relation_dim"  :   output dimensionality for each relation type,
        "max_chains"    :   maximum number of chains
        "reverse"       :   set to true to process the sequence in reverse
        "concat"        :   set to true to concatenate relations instead of attention

    compute() takes the three placeholders for the tensors described above, and optionally the
    initializer for the recurrence, and outputs
    another placeholder for the output of the layer.
    """

    def __init__(self, num_relations, input_dim, relation_dim, max_chains, 
            reverse=False, concat=False):
        self.num_relations = num_relations
        self.rdims = relation_dim
        self.input_dim = input_dim
        self.output_dim = self.num_relations*self.rdims
        self.max_chains = max_chains
        self.reverse = reverse
        self.concat = concat

        # initialize gates
        def _gate_params(name):
            gate = {}
            #h_to_h = self.rdims*self.num_relations if self.concat else self.rdims
            h_to_h = self.rdims*self.num_relations
            gate["W"] = tf.Variable(tf.random_normal((self.input_dim,self.output_dim), 
                mean=0.0, stddev=glorot(self.input_dim,self.output_dim)),
                name="W"+name, dtype=tf.float32)
            gate["U"] = tf.Variable(tf.random_normal((h_to_h,self.output_dim),
                mean=0.0, stddev=glorot(h_to_h,self.output_dim)),
                name="U"+name, dtype=tf.float32)
            gate["b"] = tf.Variable(tf.zeros((self.output_dim,)), 
                name="b"+name, dtype=tf.float32)
            return gate
        self.resetgate = _gate_params("r")
        self.updategate = _gate_params("u")
        self.hiddengate = _gate_params("h")
        self.Wstacked = tf.concat([self.resetgate["W"], self.updategate["W"],
                self.hiddengate["W"]], axis=1) # Din x 3Dout
        self.Ustacked = tf.concat([self.resetgate["U"], self.updategate["U"],
                self.hiddengate["U"]], axis=1) # Dr x 3Dout

        # initialize attention params
        if not self.concat:
            self.Watt = tf.Variable(tf.random_normal((self.num_relations,self.input_dim),
                mean=0.0, stddev=0.1),
                name="Watt", dtype=tf.float32) # Dr x Din

        # initialize initial memory state
        self.mem_init = tf.zeros((self.max_chains, self.rdims),
                                 dtype=tf.float32)

    def compute(self, X, M, Ei, Eo, Ri, Ro, init=None, mem_init=None):
        # reshape for scan
        Xre = tf.transpose(X, perm=(1,0,2))
        Mre = tf.transpose(M, perm=(1,0))
        Eire = tf.transpose(Ei, perm=(1,0,2))
        Eore = tf.transpose(Eo, perm=(1,0,2))
        Rire = tf.transpose(Ri, perm=(1,0,2))
        Rore = tf.transpose(Ro, perm=(1,0,2))

        if self.reverse:
            Xre = tf.reverse(Xre, axis=[0])
            Mre = tf.reverse(Mre, axis=[0])
            Eire = tf.reverse(Eire, axis=[0])
            Eore = tf.reverse(Eore, axis=[0])
            Rire = tf.reverse(Rire, axis=[0])
            Rore = tf.reverse(Rore, axis=[0])

        # precompute input
        Xpre = tf.tensordot(Xre, self.Wstacked, axes=[[2],[0]]) # N x B x 3Dout

        # update
        if init is None: init = tf.zeros((tf.shape(X)[0], self.output_dim), 
                dtype=tf.float32)
        if mem_init is None:
            mem_init = tf.tile(tf.expand_dims(self.mem_init, axis=0),
                               (tf.shape(X)[0], 1, 1))
        agg_init = tf.zeros((tf.shape(X)[0], self.num_relations),
                dtype = tf.float32)
        outs, mems, aggs = tf.scan(self._step, (Xre, Xpre, Mre, Eire, Eore, Rire, Rore), 
                initializer=(init,mem_init,agg_init)) # N x B x Dout

        if self.reverse:
            outs = tf.reverse(outs, axis=[0])
            mems = tf.reverse(mems, axis=[0])
            aggs = tf.reverse(aggs, axis=[0])

        return (tf.transpose(outs, perm=(1,0,2)), tf.transpose(mems, perm=(1,0,2,3)), 
                tf.transpose(aggs, perm=(1,0,2)))

    def _attention(self, x, c_r, e, r):
        EPS = 1e-100
        v = tf.tensordot(r, self.Watt, axes=[[2],[0]]) # B x C x Din
        actvs = tf.squeeze(tf.matmul(v,tf.expand_dims(x,axis=2)),axis=2) # B x C
        alphas_m = tf.exp(actvs)*e + EPS # B x C
        return alphas_m/tf.reduce_sum(alphas_m, 1, keep_dims=True) # B x C

    def _hid_prev(self, x, c_r, e, r):
        if not self.concat:
            alphas = self._attention(x, c_r, e, r) # B x C
            agg = tf.transpose(
                    tf.expand_dims(alphas, axis=2)*r, perm=[0,2,1]) # B x R x C
        else:
            agg = tf.transpose(r*tf.expand_dims(e, axis=2), 
                    perm=[0,2,1])/tf.expand_dims(
                            tf.reduce_sum(e, axis=1, keep_dims=True), axis=1) # B x R x C
        mem = tf.matmul(agg, c_r) # B x R x Dr
        return tf.reshape(mem, [-1, self.num_relations*self.rdims]), \
                tf.reduce_sum(agg, axis=2) # B x RDr

    def _step(self, prev, inps):
        hprev, mprev = prev[0], prev[1] # hprev : B x Dout, mprev : B x C x Dr
        x, xp, m, ei, eo, ri, ro = inps[0], inps[1], inps[2], inps[3], inps[4], \
                inps[5], inps[6] # x : B x Din, m : B, ei/o : B x C, ri/o : B x C

        hnew, agg = self._gru_cell(x, xp, mprev, ei, ri, self.resetgate, self.updategate,
                self.hiddengate) # B x Dout, B x R x C
        hnew_r = tf.reshape(hnew, 
                [tf.shape(x)[0], self.num_relations, self.rdims]) # B x R x Dr
        ro1hot = tf.one_hot(ro, self.num_relations, axis=2) # B x C x R
        mnew = tf.matmul(ro1hot, hnew_r) # B x C x Dr
        hnew.set_shape([None,self.output_dim])

        m_r = tf.expand_dims(m, axis=1) # B x 1
        hnew = (1.-m_r)*hprev + m_r*hnew

        eo_r = tf.expand_dims(m_r*eo, axis=2) # B x C x 1
        mnew = (1.-eo_r)*mprev + eo_r*mnew

        return hnew, mnew, agg

    def _gru_cell(self, x, xp, c, e, ri, rgate, ugate, hgate):
        def _slice(a, n):
            s = a[:,n*self.output_dim:(n+1)*self.output_dim]
            return s
        r1hot = tf.one_hot(ri, self.num_relations) # B x C x R
        prev, agg = self._hid_prev(x, c, e, r1hot) # B x RDr
        hid_to_hid = tf.matmul(prev, self.Ustacked) # B x 3Dout
        r = tf.sigmoid(_slice(xp,0) + _slice(hid_to_hid,0) + rgate["b"])
        z = tf.sigmoid(_slice(xp,1) + _slice(hid_to_hid,1) + ugate["b"])
        ht = tf.tanh(_slice(xp,2) + r*_slice(hid_to_hid,2) + hgate["b"])
        h = (1.-z)*prev + z*ht
        return h, agg
