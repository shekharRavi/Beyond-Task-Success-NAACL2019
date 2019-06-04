import os
import io
import json
import gzip
import argparse
import collections
from time import time
from nltk.tokenize import TweetTokenizer

def create_vocab(data_dir, data_file, min_occ):
    """ Creates a new vocablurary file in data_dir """
    print("Creating New Vocablurary File.")
    # Set default values
    word2i = {'<padding>': 0,
              '<start>': 1,
              '<stop>': 2,
              '<stop_dialogue>': 3,
              '<unk>': 4,
              '<yes>' : 5,
              '<no>': 6,
              '<n/a>': 7,
              }

    word2occ = collections.OrderedDict()
    categories_set = set()

    tknzr = TweetTokenizer(preserve_case=False)

    path = os.path.join(data_dir, data_file)
    with gzip.open(path) as f:
        for k , line in enumerate(f):
            dialogue = json.loads(line.decode("utf-8"))

            for qa in dialogue['qas']:
                tokens = tknzr.tokenize(qa['question'])
                for tok in tokens:
                    if tok not in word2occ:
                        word2occ[tok] = 1
                    else:
                        word2occ[tok] += 1

    for word, occ in word2occ.items():
        if occ >= min_occ and word.count('.') <= 1:
            word2i[word] = len(word2i)

    i2word = {v:k for k,v in word2i.items()}

    vocab_path = os.path.join(data_dir, 'vocab.json')
    vocab = {'word2i':word2i, 'i2word': i2word}
    with io.open(vocab_path, 'wb') as f_out:
        data = json.dumps(vocab, ensure_ascii=False)
        f_out.write(data.encode('utf8', 'replace'))

    return vocab




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-data_dir", type=str, default="data", help='Target Data Directory to store ythe vocab file')
    parser.add_argument("-data_file", type=str, default="guesswhat.train.jsonl.gz", help='Guesswhat train data file')
    parser.add_argument("-min_occ", type=int, default=3, help='Min frequency of word to be included in the vocab' )

    args = parser.parse_args()

    create_vocab(args.data_dir, args.data_file, args.min_occ)
