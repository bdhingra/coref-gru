import os
import io
import json

from collections import Counter, OrderedDict

SYMB_BEGIN = "@begin"
SYMB_END = "@end"

class Data:

    def __init__(self, dictionary, num_entities, training, validation, test,
                 word_counts, max_doc_len=0, max_qry_len=0, max_num_cand=0):
        self.dictionary = dictionary
        self.training = training
        self.validation = validation
        self.test = test
        self.vocab_size = len(dictionary[0])
        self.num_chars = len(dictionary[1]) if dictionary[1] is not None else 0
        self.num_entities = num_entities
        self.inv_dictionary = {v:k for k,v in dictionary[0].items()}
        self.word_counts = word_counts
        self.max_doc_len = max_doc_len
        self.max_qry_len = max_qry_len
        self.max_num_cand = max_num_cand

class DataPreprocessor:

    def preprocess_rc(self, params, question_dir, dictionary=None):
        """
        preprocess all data into a standalone Data object.
        """
        if dictionary is None:
            vocab_f = os.path.join(question_dir,"vocab.txt")
            word_dictionary, char_dictionary, _ = \
                    self.make_dictionary(question_dir, vocab_f, params)
            dictionary = (word_dictionary, char_dictionary)

        use_chars = params["char_dim"] > 0
        print "preparing training data ..."
        training = self.parse_json_file(question_dir + "/training.json",
                                        dictionary, use_chars,
                                        params["num_unknown_types"],
                                        params["max_chains"],
                                        params["max_word_len"],
                                        params["max_doc_len"],
                                        check_ans_in_doc=True)
        print "preparing validation data ..."
        validation = self.parse_json_file(question_dir + "/validation.json",
                                          dictionary, use_chars,
                                          params["num_unknown_types"],
                                          params["max_chains"],
                                          params["max_word_len"],
                                          params["max_doc_len"])

        data = Data(dictionary, 0, training, validation, [], None)
        return data

    def make_dictionary(self, question_dir, vocab_file, params):

        if os.path.exists(vocab_file):
            def _parse_line(line):
                s, c = line.strip().split("\t")
                return s, int(c)
            print "loading vocabularies from " + vocab_file + " ..."
            vocabularies = map(lambda x:_parse_line(x),
                               io.open(vocab_file, encoding="utf-8").readlines())
            print "loading characters from " + vocab_file + ".chars ..."
            characters = map(lambda x:_parse_line(x),
                             io.open(vocab_file + ".chars", encoding="utf-8").readlines())
        else:
            print "no " + vocab_file + " found, constructing the vocabulary list ..."

            vocab_counter = Counter()
            char_counter = Counter()

            def _add_to_vocab(filename, vocab, chars):
                for line in io.open(filename):
                    data = json.loads(line.strip())
                    for token in data["document"].split():
                        vocab[token.lower()] += 1
                        chars.update(list(token))
                    query_toks = data["query"].split()
                    vocab[query_toks[0].lower()] = 100 # to ensure this remains in vocab
                    chars.update(list(query_toks[0]))
                    for token in query_toks[1:]:
                        vocab[token.lower()] += 1
                        chars.update(list(token))

            print "reading json files ..."
            _add_to_vocab(question_dir + "/training.json", vocab_counter, char_counter)

            print "constructing vocabularies ..."
            vocabularies = vocab_counter.most_common(params["vocab_size"] -
                                                     params["num_unknown_types"])
            vocabularies += [("__unkword%d__" % ii, 0)
                             for ii in range(params["num_unknown_types"])]
            characters = char_counter.most_common(params["num_characters"] - 1)
            characters += [("__unkchar__", 0)]

            print "writing vocabularies to " + vocab_file + " ..."
            vocab_fp = io.open(vocab_file, "w", encoding="utf-8")
            vocab_fp.write('\n'.join(["%s\t%d" % (w, c) for w, c in vocabularies]))
            vocab_fp.close()
            print "writing characters to " + vocab_file + ".chars ..."
            vocab_fp = io.open(vocab_file + ".chars", "w", encoding="utf-8")
            vocab_fp.write('\n'.join(["%s\t%d" % (ch, c) for ch, c in characters]))
            vocab_fp.close()

        word_dictionary = OrderedDict([(w[0], ii) for ii, w in enumerate(vocabularies)])
        char_dictionary = OrderedDict([(w[0], ii) for ii, w in enumerate(characters)])
        print "vocab_size = %d" % len(word_dictionary)
        print "num characters = %d" % len(char_dictionary)

        return word_dictionary, char_dictionary, 0

    @staticmethod
    def process_question(doc_raw, doc_lower, qry_raw, qry_lower, ans_raw, ans_lower,
                         cand_raw, cand_lower, w_dict, c_dict, use_chars, fname,
                         num_unks, max_word_len, wrapqry=True):
        # wrap the query with special symbols
        if wrapqry:
            qry_raw.insert(0, SYMB_BEGIN)
            qry_raw.append(SYMB_END)

        # find OOV tokens
        oov = set()
        def _add_to_oov(tokens):
            for token in tokens:
                if token not in w_dict: oov.add(token)
        _add_to_oov(doc_lower)
        _add_to_oov(qry_lower)
        for cand in cand_lower: _add_to_oov(cand)
        # assign unk embeddings to oov tokens
        unk_dict = {}
        for ii, token in enumerate(oov):
            unk_dict[token] = w_dict[u"__unkword%d__" % (ii % num_unks)]

        # tokens/entities --> indexes
        def _map_token(token):
            if token in unk_dict: return unk_dict[token]
            else: return w_dict[token]
        doc_words = map(lambda w:_map_token(w), doc_lower)
        qry_words = map(lambda w:_map_token(w), qry_lower)
        ans = map(lambda w:_map_token(w), ans_lower)
        cand = [map(lambda w:_map_token(w), c) for c in cand_lower]

        # tokens --> character index lists
        if use_chars:
            doc_chars = map(lambda w: map(lambda c:c_dict.get(c,c_dict[u"__unkchar__"]), 
                list(w)[:max_word_len]), doc_raw)
            qry_chars = map(lambda w: map(lambda c:c_dict.get(c,c_dict[u"__unkchar__"]), 
                list(w)[:max_word_len]), qry_raw)
        else:
            doc_chars, qry_chars = [], []

        return doc_words, qry_words, ans, cand, doc_chars, qry_chars

    def parse_json_file(self, filename, dictionary, use_chars, num_unks,
                        max_chains, max_word_len, max_doc_len,
                        check_ans_in_doc=False):
        w_dict, c_dict = dictionary[0], dictionary[1]
        questions = []
        max_qry_len, max_cands, max_ents = 0, 0, 0
        with io.open(filename) as f:
            for ni, line in enumerate(f):
                # read
                data = json.loads(line.rstrip())
                doc_raw = data["document"].split()[:max_doc_len]
                qry_raw = data["query"].split()
                ans_raw = data["answer"].split()
                cand_raw = data["candidates"]
                annotations = data["annotations"]
                ii = data["id"]
                mentions = data["mentions"]
                corefs = data["coref_onehot"][:max_chains-1]

                # checks
                assert ans_raw in cand_raw
                doc_lower = [t.lower() for t in doc_raw]
                qry_lower = [t.lower() for t in qry_raw]
                ans_lower = [t.lower() for t in ans_raw]
                cand_lower = [[t.lower() for t in cand] for cand in cand_raw]
                if not any(aa in doc_lower for aa in ans_lower):
                    if check_ans_in_doc:
                        raise ValueError((data["document"].split()[max_doc_len:], ans_raw))
                    else:
                        print "answer not in doc %s" % ii
                        continue

                # stats
                if len(qry_raw) > max_qry_len: max_qry_len = len(qry_raw)
                if len(cand_raw) > max_cands: max_cands = len(cand_raw)
                if len(corefs) > max_ents: max_ents = len(corefs)

                questions.append(
                    self.process_question(doc_raw, doc_lower, qry_raw, qry_lower,
                                          ans_raw, ans_lower, cand_raw, cand_lower,
                                          w_dict, c_dict, use_chars, ii, num_unks,
                                          max_word_len, wrapqry=False) +
                    (corefs, mentions, annotations, ii))
        print("retained %d out of %d questions" % (len(questions), ni + 1))
        print("maximum query length = %d" % max_qry_len)
        print("maximum number of candidates = %d" % max_cands)
        print("maximum number of tracked entities = %d" % max_ents)
        return questions
