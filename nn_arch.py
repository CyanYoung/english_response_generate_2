from keras.layers import Conv1D, GlobalMaxPooling1D, Dense
from keras.layers import Dropout, Multiply, Concatenate, RepeatVector

import keras.backend as K
from keras.engine.topology import Layer


win_len = 7
seq_len = 100


def cnn_s2s(embed_input1, embed_input2, vocab_num):
    h1_n = s2s_encode(embed_input1)
    return s2s_decode(embed_input2, h1_n, vocab_num)


def s2s_encode(x1):
    conv = Conv1D(filters=128, kernel_size=win_len, padding='valid', name='conv')
    gate = Conv1D(filters=128, kernel_size=win_len, padding='valid', activation='sigmoid', name='gate')
    mp = GlobalMaxPooling1D()
    da = Dense(200, activation='relu', name='encode')
    x1 = conv(x1)
    g = gate(x1)
    h1 = Multiply()([x1, g])
    h1_n = mp(h1)
    return da(h1_n)


def s2s_decode(x2, h1_n, vocab_num):
    conv = Conv1D(filters=128, kernel_size=win_len, padding='valid', name='conv')
    gate = Conv1D(filters=128, kernel_size=win_len, padding='valid', activation='sigmoid', name='gate')
    da = Dense(vocab_num, activation='softmax', name='classify')
    x2 = conv(x2)
    g = gate(x2)
    h2 = Multiply()([x2, g])
    h1_n = RepeatVector(seq_len)(h1_n)
    s2 = Concatenate()([h2, h1_n])
    s2 = Dropout(0.2)(s2)
    return da(s2)


class Attend(Layer):
    def __init__(self, unit, **kwargs):
        self.unit = unit
        super(Attend, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        self.seq_len = input_shape[0][1]
        self.embed_len = input_shape[0][2]
        self.w = self.add_weight(name='w', shape=(self.embed_len * 2, self.unit),
                                 initializer='glorot_uniform')
        self.b1 = self.add_weight(name='b1', shape=(self.unit,),
                                  initializer='zeros')
        self.v = self.add_weight(name='v', shape=(self.unit, 1),
                                 initializer='glorot_uniform')
        self.b2 = self.add_weight(name='b2', shape=(1,), initializer='zeros')
        super(Attend, self).build(input_shape)

    def call(self, x):
        assert isinstance(x, list)
        h1, h2 = x
        c = list()
        for i in range(self.seq_len):
            h2_i = K.repeat(h2[:, i, :], self.seq_len)
            x = K.concatenate([h1, h2_i])
            p = K.tanh(K.dot(x, self.w) + self.b1)
            p = K.softmax(K.dot(p, self.v) + self.b2)
            p = K.squeeze(p, axis=-1)
            p = K.repeat(p, self.embed_len)
            p = K.permute_dimensions(p, (0, 2, 1))
            c_i = K.sum(p * h1, axis=1, keepdims=True)
            c.append(c_i)
        return K.concatenate(c, axis=1)

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        return input_shape[0]


def cnn_att(embed_input1, embed_input2, vocab_num):
    h1 = att_encode(embed_input1)
    return att_decode(embed_input2, h1, vocab_num)


def att_encode(x1):
    conv = Conv1D(filters=128, kernel_size=win_len, padding='valid', name='conv')
    gate = Conv1D(filters=128, kernel_size=win_len, padding='valid', activation='sigmoid', name='gate')
    da = Dense(200, activation='relu', name='encode')
    x1 = conv(x1)
    g = gate(x1)
    h1 = Multiply()([x1, g])
    return da(h1)


def att_decode(x2, h1, vocab_num):
    conv = Conv1D(filters=128, kernel_size=win_len, padding='valid', name='conv')
    gate = Conv1D(filters=128, kernel_size=win_len, padding='valid', activation='sigmoid', name='gate')
    attend = Attend(200, name='attend')
    da = Dense(vocab_num, activation='softmax', name='classify')
    x2 = conv(x2)
    g = gate(x2)
    h2 = Multiply()([x2, g])
    c = attend([h1, h2])
    s2 = Concatenate()([h2, c])
    s2 = Dropout(0.2)(s2)
    return da(s2)
