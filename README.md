## English Dialogue Generate 2018-11

#### 1.preprocess

prepare() 将数据保存为 (text1, text2) 格式，打乱后划分训练、测试集

#### 2.represent

add_flag() 为 text2 添加控制符、shift() 分别删去 bos、eos 得到 sent2、label

tokenize() 通过 text1 和 flag_text2 建立词索引、构造 embed_mat

align() 对训练数据 sent1 的尾部，sent2、label 的头部，填充或截取为定长序列

#### 3.build

s2s 编码器返回最后状态 h1_n、作为解码器初始状态，att 编码器返回所有状态 h1

Attend() 通过解码器各状态 h2_i 与 h1 返回语境向量 c_i，h2_i 与 c_i 共同决定输出

#### 4.generate

predict() 先对输入进行编码、再通过采样或搜索进行生成，check() 忽略无效词

#### 5.eval

通过搜索进行生成，使用 bleu 评价质量、即 n-gram 的平均重合度