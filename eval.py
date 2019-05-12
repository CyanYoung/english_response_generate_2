import pickle as pk

import numpy as np

from generate import models

from util import map_item


path_sent1 = 'feat/sent1_test.pkl'
path_sent2 = 'feat/sent2_test.pkl'
path_label = 'feat/label_test.pkl'
with open(path_sent1, 'rb') as f:
    sent1s = pk.load(f)
with open(path_sent2, 'rb') as f:
    sent2s = pk.load(f)
with open(path_label, 'rb') as f:
    labels = pk.load(f)


def test(name, sent1s, labels):
    encode = map_item(name + '_encode', models)
    states = encode.predict(sent1s)
    decode = map_item(name + '_decode', models)
    probs = decode.predict([sent2s, states])
    len_sum, log_sum = [0] * 2
    for sent2, label, prob in zip(sent2s, labels, probs):
        bound = sum(sent2 > 0)
        len_sum = len_sum + bound
        sent_log = 0
        for i in range(bound):
            sent_log = sent_log + np.log(prob[i][label[i]])
        log_sum = log_sum + sent_log
    print('\n%s %s %.2f' % (name, 'perp:', np.power(2, -log_sum / len_sum)))


if __name__ == '__main__':
    test('s2s', sent1s, labels)
    test('att', sent1s, labels)
