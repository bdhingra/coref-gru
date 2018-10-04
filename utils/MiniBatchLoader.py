import numpy as np
import random

class MiniBatchLoader(object):

    def __init__(self, params, questions, batch_size, shuffle=True,
            ensure_answer=True):
        self.batch_size = batch_size
        self.questions = questions
        self.bins = self.build_bins(self.questions)
        self.max_word_len = params["max_word_len"]
        self.max_chains = params["max_chains"]
        self.max_doc_len = params["max_doc_len"]
        self.shuffle = shuffle
        self.ensure_answer = ensure_answer
	self.reset()

    def __iter__(self):
        """make the object iterable"""
        return self

    def build_bins(self, questions):
        """
        returns a dictionary
            key: document length (rounded to the powers of two)
            value: indexes of questions with document length equal to key
        """
        # round the input to the nearest power of two
        round_to_power = lambda x: 2**(int(np.log2(x-1))+1) if x>1 else 1
        # round to the nearest multiple of 200
        round_to_multiple = lambda x: 200 * (int((x-1)/200) + 1)

        doc_len = map(lambda x:round_to_multiple(len(x[0])), questions)
        bins = {}
        for i, l in enumerate(doc_len):
            if l not in bins:
                bins[l] = []
            bins[l].append(i)

        print "using following bin sizes: ", " ".join("(%d, %d)" % (k, len(v))
                                                      for k, v in bins.iteritems())

        return bins

    def reset(self):
        """new iteration"""
        self.ptr = 0

        # randomly shuffle the question indices in each bin
        if self.shuffle:
            for ixs in self.bins.itervalues():
                random.shuffle(ixs)

        # construct a list of mini-batches where each batch is a list of question indices
        # questions within the same batch have identical max document length 
        self.batch_pool = []
        for l, ixs in self.bins.iteritems():
            n = len(ixs)
            k = n/self.batch_size if n % self.batch_size == 0 else n/self.batch_size+1
            ixs_list = [(ixs[self.batch_size*i:min(n, self.batch_size*(i+1))],l) for i in range(k)]
            self.batch_pool += ixs_list

        # randomly shuffle the mini-batches
        if self.shuffle:
            random.shuffle(self.batch_pool)


class MiniBatchLoaderMention(MiniBatchLoader):

    def next(self):
        """load the next batch"""
        if self.ptr == len(self.batch_pool):
            self.reset()
            raise StopIteration()

        ixs = self.batch_pool[self.ptr][0]
        curr_max_doc_len = self.batch_pool[self.ptr][1]
        curr_batch_size = len(ixs)
        curr_max_qry_len, curr_max_cands, curr_max_ents, curr_max_mens = 0, 0, 0, 0
        for ix in ixs:
            qry_len = len(self.questions[ix][1])
            num_cands = len(self.questions[ix][3])
            num_ents = len(self.questions[ix][6])
            num_mens = len(self.questions[ix][7])
            if qry_len > curr_max_qry_len: curr_max_qry_len = qry_len
            if num_cands > curr_max_cands: curr_max_cands = num_cands
            if num_ents > curr_max_ents: curr_max_ents = num_ents
            if num_mens > curr_max_mens: curr_max_mens = num_mens

        dw = np.zeros((curr_batch_size, curr_max_doc_len), dtype='int32') # document words
        m_dw = np.zeros((curr_batch_size, curr_max_doc_len), dtype='float32')  # document word mask
        qw = np.zeros((curr_batch_size, curr_max_qry_len), dtype='int32') # query words
        m_qw = np.zeros((curr_batch_size, curr_max_qry_len), dtype='float32')  # query word mask

        dc = np.zeros((curr_batch_size, curr_max_doc_len, self.max_word_len), dtype="int32")
        m_dc = np.zeros((curr_batch_size, curr_max_doc_len, self.max_word_len), dtype="float32")
        qc = np.zeros((curr_batch_size, curr_max_qry_len, self.max_word_len), dtype="int32")
        m_qc = np.zeros((curr_batch_size, curr_max_qry_len, self.max_word_len), dtype="float32")

        cd = np.zeros((curr_batch_size, curr_max_doc_len, curr_max_cands), 
                dtype='int32')   # candidate answers
        m_cd = np.zeros((curr_batch_size, curr_max_doc_len), dtype='float32') # candidate mask

        edges_in = np.zeros((curr_batch_size, curr_max_doc_len, self.max_chains), dtype="float32")
        edges_out = np.zeros((curr_batch_size, curr_max_doc_len, self.max_chains), dtype="float32")
        edges_in[:, :, 0] = 1.
        edges_out[:, :, 0] = 1.

        a = np.zeros((curr_batch_size, ), dtype='int32')    # correct answer
        fnames = ['']*curr_batch_size
        annots = []

        for n, ix in enumerate(ixs):

            doc_w, qry_w, ans, cand, doc_c, qry_c, corefs, mentions, annotations, fname = self.questions[ix]

            # document and query
            dw[n, :len(doc_w)] = doc_w
            qw[n, :len(qry_w)] = qry_w
            m_dw[n, :len(doc_w)] = 1
            m_qw[n, :len(qry_w)] = 1
            for t in range(len(doc_c)):
                dc[n, t, :len(doc_c[t])] = doc_c[t]
                m_dc[n, t, :len(doc_c[t])] = 1
            for t in range(len(qry_c)):
                qc[n, t, :len(qry_c[t])] = qry_c[t]
                m_qc[n, t, :len(qry_c[t])] = 1

            # search candidates in doc
            found_answer = False
            for it, cc in enumerate(cand):
                index = [ii for ii in range(len(doc_w)) if doc_w[ii] in cc]
                m_cd[n, index] = 1
                cd[n, index, it] = 1
                if ans == cc: 
                    found_answer = True
                    a[n] = it # answer
                    if self.ensure_answer: assert index, fname
            assert found_answer

            # graph edges
            for ic, chain in enumerate(corefs):
                for item in chain:
                    if item[2] != -1:
                        if mentions[item[2]][0] < curr_max_doc_len:
                            edges_in[n, mentions[item[2]][0], ic+1] = 1.
                    if item[0] != -1:
                        if mentions[item[0]][1]-1 < curr_max_doc_len:
                            edges_out[n, mentions[item[0]][1]-1, ic+1] = 1.

            annots.append(annotations)
            fnames[n] = fname

        self.ptr += 1

        return dw, m_dw, qw, m_qw, dc, m_dc, qc, m_qc, cd, m_cd, a, (edges_in, edges_out), annots, fnames
