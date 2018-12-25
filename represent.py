import json
import pickle as pk

import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


embed_len = 200
max_vocab = 10000
win_len = 5
seq_len = 100

bos, eos = '<', '>'

path_word_vec = 'feat/word_vec.pkl'
path_word2ind = 'model/word2ind.pkl'
path_embed = 'feat/embed.pkl'


def load(path):
    with open(path, 'rb') as f:
        item = pk.load(f)
    return item


def save(item, path):
    with open(path, 'wb') as f:
        pk.dump(item, f)


def add_flag(texts):
    flag_texts = list()
    for text in texts:
        flag_texts.append(' '.join([bos, text, eos]))
    return flag_texts


def shift(flag_texts):
    sents = [text[:-2] for text in flag_texts]
    labels = [text[2:] for text in flag_texts]
    return sents, labels


def tokenize(texts, path_word2ind):
    model = Tokenizer(num_words=max_vocab, filters='', lower=True, oov_token='oov')
    model.fit_on_texts(texts)
    save(model, path_word2ind)


def embed(path_word2ind, path_word_vec, path_embed):
    model = load(path_word2ind)
    word_inds = model.word_index
    word_vecs = load(path_word_vec)
    vocab = word_vecs.keys()
    vocab_num = min(max_vocab + 1, len(word_inds) + 1)
    embed_mat = np.zeros((vocab_num, embed_len))
    for word, ind in word_inds.items():
        if word in vocab:
            if ind < max_vocab:
                embed_mat[ind] = word_vecs[word]
    save(embed_mat, path_embed)


def add_buf(seqs, sym):
    buf = [0] * int((win_len - 1) / 2)
    buf_seqs = list()
    for seq in seqs.tolist():
        if sym:
            buf_seqs.append(buf + seq + buf)
        else:
            buf_seqs.append(buf * 2 + seq)
    return np.array(buf_seqs)


def align(sents, path_sent, phase, extra):
    model = load(path_word2ind)
    seqs = model.texts_to_sequences(sents)
    loc = 'post' if phase == 'decode' else 'pre'
    pad_seqs = pad_sequences(seqs, maxlen=seq_len, padding=loc, truncating=loc)
    if extra:
        sym = False if phase == 'decode' else True
        pad_seqs = add_buf(pad_seqs, sym)
    save(pad_seqs, path_sent)


def vectorize(paths, mode):
    with open(paths['data'], 'r') as f:
        pairs = json.load(f)
    text1s, text2s = zip(*pairs)
    text1s, text2s = list(text1s), list(text2s)
    if mode == 'train':
        flag_text2s = add_flag(text2s)
        tokenize(text1s + flag_text2s, path_word2ind)
        embed(path_word2ind, path_word_vec, path_embed)
        sent2s, labels = shift(flag_text2s)
        align(text1s, paths['sent1'], 'encode', extra=True)
        align(sent2s, paths['sent2'], 'decode', extra=True)
        align(labels, paths['label'], 'decode', extra=False)
    else:
        save(text1s, paths['sent1'])
        save(text2s, paths['label'])


if __name__ == '__main__':
    paths = dict()
    paths['data'] = 'data/train.json'
    paths['sent1'] = 'feat/sent1_train.pkl'
    paths['sent2'] = 'feat/sent2_train.pkl'
    paths['label'] = 'feat/label_train.pkl'
    vectorize(paths, 'train')
    paths['data'] = 'data/test.json'
    paths['sent1'] = 'feat/sent1_test.pkl'
    paths['label'] = 'feat/label_test.pkl'
    vectorize(paths, 'test')
