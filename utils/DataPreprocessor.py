import os
import json
from itertools import izip

MAX_WORD_LEN = 10

SYMB_BEGIN = "@begin"
SYMB_END = "@end"

class Data:

    def __init__(self, dictionary, num_entities, training, validation, test):
        self.dictionary = dictionary
        self.training = training
        self.validation = validation
        self.test = test
        self.vocab_size = len(dictionary[0])
        self.num_chars = len(dictionary[1])
        self.num_entities = num_entities
        self.inv_dictionary = {v:k for k,v in dictionary[0].items()}
        if training:
            self.max_num_cand = max(map(lambda x:len(x[3]), 
                training+validation+test))
        else:
            self.max_num_cand = max(map(lambda x:len(x[3]), 
                validation+test))

class DataPreprocessor:

    def preprocess(self, question_dir, max_chains=100, 
                   no_training_set=False, use_chars=True):
        """
        preprocess all data into a standalone Data object.
        the training set will be left out (to save debugging time) when no_training_set
        is True.
        """
        vocab_f = os.path.join(question_dir,"vocab.txt")
        word_dictionary, char_dictionary, num_entities = \
                self.make_dictionary(question_dir, vocab_file=vocab_f)
        dictionary = (word_dictionary, char_dictionary)
        if no_training_set:
            training = None
        else:
            print "preparing training data ..."
            training = self.parse_all_files(question_dir + "/training", 
                                            dictionary, use_chars, max_chains)
        print "preparing validation data ..."
        validation = self.parse_all_files(question_dir + "/validation", 
                                          dictionary, use_chars, max_chains)
        print "preparing test data ..."
        test = self.parse_all_files(question_dir + "/test", 
                                    dictionary, use_chars, max_chains)

        data = Data(dictionary, num_entities, training, validation, test)
        return data

    def make_dictionary(self, question_dir, vocab_file):

        if os.path.exists(vocab_file):
            print "loading vocabularies from " + vocab_file + " ..."
            vocabularies = map(lambda x:x.strip(), open(vocab_file).readlines())
        else:
            print "no " + vocab_file + " found, constructing the vocabulary list ..."

            vocab_set = set()

            f = open(question_dir+'/training.data')
            for line in f:
                vocab_set |= set(line.rstrip().split())
            f.close()
            f = open(question_dir+'/validation.data')
            for line in f:
                vocab_set |= set(line.rstrip().split())
            f.close()
            f = open(question_dir+'/test.data')
            for line in f:
                vocab_set |= set(line.rstrip().split())
            f.close()

            vocab_set.add(SYMB_BEGIN)
            vocab_set.add(SYMB_END)
            vocabularies = list(vocab_set)

            print "writing vocabularies to " + vocab_file + " ..."
            vocab_fp = open(vocab_file, "w")
            vocab_fp.write('\n'.join(vocabularies))
            vocab_fp.close()

        vocab_size = len(vocabularies)
        word_dictionary = dict(zip(vocabularies, range(vocab_size)))
        char_set = set([c for w in vocabularies for c in list(w)])
        char_set.add(' ')
        char_dictionary = dict(zip(list(char_set), range(len(char_set))))
        num_entities = len([v for v in vocabularies if v.startswith('@entity')])
        print "vocab_size = %d" % vocab_size
        print "num characters = %d" % len(char_set)
        print "%d anonymoused entities" % num_entities
        print "%d other tokens (including @placeholder, %s and %s)" % (
                vocab_size-num_entities, SYMB_BEGIN, SYMB_END)

        return word_dictionary, char_dictionary, num_entities

    @staticmethod
    def process_question(doc_raw, qry_raw, ans_raw, cand_raw, w_dict, 
            c_dict, use_chars, fname):
        # wrap the query with special symbols
        qry_raw.insert(0, SYMB_BEGIN)
        qry_raw.append(SYMB_END)
        try:
            cloze = qry_raw.index('@placeholder')
        except ValueError:
            try:
                at = qry_raw.index('@')
                print '@placeholder not found in ', fname, '. Fixing...'
                qry_raw = qry_raw[:at] + [''.join(qry_raw[at:at+2])] + qry_raw[at+2:]
                cloze = qry_raw.index('@placeholder')
            except ValueError:
                cloze = -1

        # tokens/entities --> indexes
        doc_words = map(lambda w:w_dict[w], doc_raw)
        qry_words = map(lambda w:w_dict[w], qry_raw)
        if use_chars:
            doc_chars = map(lambda w:map(lambda c:c_dict.get(c,c_dict[' ']), 
                list(w)[:MAX_WORD_LEN]), doc_raw)
            qry_chars = map(lambda w:map(lambda c:c_dict.get(c,c_dict[' ']), 
                list(w)[:MAX_WORD_LEN]), qry_raw)
        else:
            doc_chars, qry_chars = [], []
        ans = map(lambda w:w_dict.get(w,0), ans_raw.split())
        cand = [map(lambda w:w_dict.get(w,0), c) for c in cand_raw]

        return doc_words, qry_words, ans, cand, doc_chars, qry_chars, cloze

    def parse_data_file(self, fdata, frels, dictionary, use_chars, stops, max_chains):
        """
        parse a *.data file into list of tuple(document, query, answer, filename)
        """
        w_dict, c_dict = dictionary[0], dictionary[1]
        questions = []
        with open(fdata) as data, open(frels) as rels:
            for ii, (raw, chains) in enumerate(izip(data, rels)):
                sents = raw.rstrip().rsplit(' . ', 1) # doc and query
                doc_raw = sents[0].split()+['.'] # document
                qry_tok = sents[1].rstrip().split()
                qry_raw, ans_raw =  qry_tok[:-1], qry_tok[-1] # query and answer
                cand_raw = filter(lambda x:x not in stops, set(doc_raw))
                if ans_raw not in cand_raw: continue
                cand_raw = [[cd] for cd in cand_raw]

                coref = json.loads(chains)[:max_chains-1]
                if not any(aa in doc_raw for aa in ans_raw.split()):
                    print "answer not in doc %s" % ii
                    continue

                questions.append(self.process_question(doc_raw, qry_raw, ans_raw, 
                    cand_raw, w_dict, c_dict, use_chars, ii) + (coref,ii))

        return questions

    def parse_all_files(self, directory, dictionary, use_chars, max_chains):
        """
        parse all files under the given directory into a list of questions,
        where each element is in the form of (document, query, answer, filename)
        """
        basedir = directory.rsplit('/',1)[0]
        stops = open('utils/shortlist-stopwords.txt').read().splitlines()
        questions = self.parse_data_file(directory+'.data', 
                directory+'.coref_onehot',
                dictionary, use_chars, stops, max_chains)
        return questions
