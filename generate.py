import pickle as pk

import re

import numpy as np
from numpy.random import choice

from keras.models import Model
from keras.layers import Input, Embedding

from keras.preprocessing.sequence import pad_sequences

from preprocess import clean

from nn_arch import s2s_encode, s2s_decode, att_encode, att_decode

from util import load_word_re, map_item


def define_model(name, embed_mat, seq_len, mode):
    vocab_num, embed_len = embed_mat.shape
    embed = Embedding(input_dim=vocab_num, output_dim=embed_len, input_length=seq_len, name='embed')
    input = Input(shape=(seq_len,))
    if name == 'att':
        state = Input(shape=(seq_len, embed_len))
    else:
        state = Input(shape=(embed_len,))
    embed_input = embed(input)
    arch = map_item('_'.join([name, mode]), archs)
    if mode == 'decode':
        output = arch(embed_input, state, vocab_num)
        return Model([input, state], output)
    else:
        output = arch(embed_input)
        return Model(input, output)


def load_model(name, embed_mat, seq_len, mode):
    model = define_model(name, embed_mat, seq_len, mode)
    model.load_weights(map_item(name, paths), by_name=True)
    return model


def ind2word(word_inds):
    ind_words = dict()
    for word, ind in word_inds.items():
        ind_words[ind] = word
    return ind_words


def check(probs, cand, keep_eos):
    max_probs, max_inds = list(), list()
    sort_probs = -np.sort(-probs)
    sort_inds = np.argsort(-probs)
    for prob, ind in zip(list(sort_probs), list(sort_inds)):
        if not keep_eos and ind == eos_ind:
            continue
        if ind not in skip_inds:
            max_probs.append(prob)
            max_inds.append(ind)
        if len(max_probs) == cand:
            break
    return max_probs, max_inds


def sample(decode, state, cand):
    sent2 = bos
    next_word, count = '', 0
    while next_word != eos and count < max_len:
        count = count + 1
        sent2 = ' '.join([sent2, next_word])
        seq2 = word2ind.texts_to_sequences([sent2])[0]
        pad_seq2 = pad_sequences([seq2], maxlen=seq_len, padding='post', truncating='post')
        step = min(count - 1, seq_len - 1)
        probs = decode.predict([pad_seq2, state])[0][step]
        max_probs, max_inds = check(probs, cand, keep_eos=True)
        if max_inds[0] == word_inds[eos]:
            next_word = eos
        else:
            max_probs = max_probs / np.sum(max_probs)
            next_word = ind_words[choice(max_inds, p=max_probs)]
    return sent2[3:]


def search(decode, state, cand):
    bos_ind = [word_inds[bos]]
    pad_bos = pad_sequences([bos_ind], maxlen=seq_len, padding='post', truncating='post')
    logs = np.log(decode.predict([pad_bos, state])[0][0])
    max_logs, max_inds = check(logs, cand, keep_eos=False)
    sent2s, log_sums = [bos] * cand, max_logs
    fin_sent2s, fin_logs = list(), list()
    next_words, count = [ind_words[ind] for ind in max_inds], 1
    while cand > 0:
        log_mat, ind_mat = list(), list()
        count = count + 1
        for i in range(cand):
            sent2s[i] = ' '.join([sent2s[i], next_words[i]])
            seq2 = word2ind.texts_to_sequences([sent2s[i]])[0]
            pad_seq2 = pad_sequences([seq2], maxlen=seq_len, padding='post', truncating='post')
            step = min(count - 1, seq_len - 1)
            logs = np.log(decode.predict([pad_seq2, state])[0][step])
            max_logs, max_inds = check(logs, cand, keep_eos=True)
            max_logs = max_logs + log_sums[i]
            log_mat.append(max_logs)
            ind_mat.append(max_inds)
        max_logs = -np.sort(-np.array(log_mat), axis=None)[:cand]
        next_sent2s, next_words, log_sums = list(), list(), list()
        for log in max_logs:
            args = np.where(log_mat == log)
            sent_arg, ind_arg = int(args[0][0]), int(args[1][0])
            next_word = ind_words[ind_mat[sent_arg][ind_arg]]
            if next_word != eos and count < max_len:
                next_words.append(next_word)
                next_sent2s.append(sent2s[sent_arg])
                log_sums.append(log)
            else:
                cand = cand - 1
                fin_sent2s.append(sent2s[sent_arg])
                fin_logs.append(log / count)
        sent2s = next_sent2s
    max_arg = np.argmax(np.array(fin_logs))
    return fin_sent2s[max_arg][2:]


max_len = 50
seq_len = 100

bos, eos = '*', '#'

path_stop_word = 'dict/stop_word.txt'
stop_word_re = load_word_re(path_stop_word)

path_embed = 'feat/embed.pkl'
path_word2ind = 'model/word2ind.pkl'
with open(path_embed, 'rb') as f:
    embed_mat = pk.load(f)
with open(path_word2ind, 'rb') as f:
    word2ind = pk.load(f)
word_inds = word2ind.word_index

eos_ind = word_inds[eos]
skip_inds = [0, word_inds['oov']]

ind_words = ind2word(word_inds)

archs = {'s2s_encode': s2s_encode,
         's2s_decode': s2s_decode,
         'att_encode': att_encode,
         'att_decode': att_decode}

funcs = {'sample': sample,
         'search': search}

paths = {'s2s': 'model/s2s.h5',
         'att': 'model/att.h5'}

models = {'s2s_encode': load_model('s2s', embed_mat, seq_len, 'encode'),
          's2s_decode': load_model('s2s', embed_mat, seq_len, 'decode'),
          'att_encode': load_model('att', embed_mat, seq_len, 'encode'),
          'att_decode': load_model('att', embed_mat, seq_len, 'decode')}


def predict(text, name, mode):
    sent1 = ' '.join([text, eos])
    sent1 = re.sub(stop_word_re, '', sent1)
    seq1 = word2ind.texts_to_sequences([sent1])[0]
    pad_seq1 = pad_sequences([seq1], maxlen=seq_len, padding='pre', truncating='pre')
    encode = map_item(name + '_encode', models)
    state = encode.predict(pad_seq1)
    decode = map_item(name + '_decode', models)
    func = map_item(mode, funcs)
    return func(decode, state, cand=5)


if __name__ == '__main__':
    while True:
        text = input('text: ')
        text = clean(text)
        print('s2s: %s' % predict(text, 's2s', 'search'))
        print('att: %s' % predict(text, 'att', 'search'))
        print('s2s: %s' % predict(text, 's2s', 'sample'))
        print('att: %s' % predict(text, 'att', 'sample'))
