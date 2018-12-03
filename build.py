import pickle as pk

import numpy as np

from keras.models import Model
from keras.layers import Input, Embedding
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

from nn_arch import s2s, att

from util import map_item


batch_size = 128

path_embed = 'feat/embed.pkl'
with open(path_embed, 'rb') as f:
    embed_mat = pk.load(f)

funcs = {'s2s': s2s,
         'att': att}

paths = {'s2s': 'model/s2s.h5',
         'att': 'model/att.h5'}


def load_feat(path_feats):
    with open(path_feats['sent1'], 'rb') as f:
        sent1s = pk.load(f)
    with open(path_feats['sent2'], 'rb') as f:
        sent2s = pk.load(f)
    with open(path_feats['label'], 'rb') as f:
        labels = pk.load(f)
    return sent1s, sent2s, labels


def compile(name, embed_mat, seq_len):
    vocab_num, embed_len = embed_mat.shape
    embed = Embedding(input_dim=vocab_num, output_dim=embed_len,
                      weights=[embed_mat], input_length=seq_len, trainable=True, name='embed')
    input1 = Input(shape=(seq_len,))
    input2 = Input(shape=(seq_len,))
    embed_input1 = embed(input1)
    embed_input2 = embed(input2)
    func = map_item(name, funcs)
    output = func(embed_input1, embed_input2, vocab_num)
    model = Model([input1, input2], output)
    model.summary()
    model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
    return model


def fit(name, epoch, embed_mat, path_feats):
    sent1s, sent2s, labels = load_feat(path_feats)
    seq_len = len(sent1s[0])
    model = compile(name, embed_mat, seq_len)
    check_point = ModelCheckpoint(map_item(name, paths), monitor='val_loss', verbose=True, save_best_only=True)
    labels = np.expand_dims(labels, -1)
    model.fit([sent1s, sent2s], labels, batch_size=batch_size, epochs=epoch,
              verbose=True, callbacks=[check_point], validation_split=0.2)


if __name__ == '__main__':
    path_feats = dict()
    path_feats['sent1'] = 'feat/sent1_train.pkl'
    path_feats['sent2'] = 'feat/sent2_train.pkl'
    path_feats['label'] = 'feat/label_train.pkl'
    fit('s2s', 10, embed_mat, path_feats)
    fit('att', 10, embed_mat, path_feats)
