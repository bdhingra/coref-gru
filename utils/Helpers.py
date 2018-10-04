import numpy as np

EMBED_DIM=128

def load_word2vec_embeddings(dictionary, vocab_embed_file):
    if vocab_embed_file is None: return None, EMBED_DIM

    fp = open(vocab_embed_file)

    info = fp.readline().split()
    embed_dim = int(info[1])

    vocab_embed = {}
    for line in fp:
        line = line.split()
        vocab_embed[line[0]] = np.array(map(float, line[1:]), dtype='float32')
    fp.close()

    vocab_size = len(dictionary)
    W = np.random.randn(vocab_size, embed_dim).astype('float32')
    n = 0
    for w, i in dictionary.iteritems():
        if w in vocab_embed:
            W[i,:] = vocab_embed[w]
            n += 1
    print "%d/%d vocabs are initialized with word2vec embeddings." % (n, vocab_size)
    return W, embed_dim
