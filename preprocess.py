import json

import re

from random import shuffle

from util import load_word_re


path_stop_word = 'dict/stop_word.txt'
stop_word_re = load_word_re(path_stop_word)


def save(path, pairs):
    with open(path, 'w') as f:
        json.dump(pairs, f, ensure_ascii=False, indent=4)


def prepare(path_univ, path_train, path_test):
    pairs = list()
    with open(path_univ, 'r') as f:
        for line in f:
            text1, text2 = line.strip().split('\t')
            text1 = re.sub(stop_word_re, '', text1)
            text2 = re.sub(stop_word_re, '', text2)
            pairs.append((text1, text2))
    shuffle(pairs)
    bound = int(len(pairs) * 0.9)
    save(path_train, pairs[:bound])
    save(path_test, pairs[bound:])


if __name__ == '__main__':
    path_univ = 'data/univ.txt'
    path_train = 'data/train.json'
    path_test = 'data/test.json'
    prepare(path_univ, path_train, path_test)