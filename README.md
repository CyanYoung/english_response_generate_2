## English Dialogue Generate 2018-11

#### 1.preprocess

prepare() 将数据保存为 (text1, text2) 格式，打乱后划分训练、测试集

#### 2.represent

add_flag() 添加控制符、shift() 对 text2 分别删去 bos、eos 得到 sent2、label

tokenize() 通过 sent1 和 flag_text2 建立词索引、构造 embed_mat

align() 对训练数据 sent1 头部，sent2、label 尾部，填充或截取为定长序列

add_buf() 再对 sent1 头部、尾部进行单倍，sent2 头部进行双倍填充，对齐 label

#### 3.build

通过 cnn 构建语言生成模型，s2s 编码器对词特征 max_pool 返回句特征 h1_n 

分别与解码器输入连接，att 编码器返回词特征 h1，解码器返回词特征 h2

Attend() 比较 h2_i 与 h1、对 h1 加权平均返回语境 c_i，h2_i 与 c_i 共同决定输出

#### 4.generate

predict() 先对输入进行编码、再通过采样或搜索进行解码，check() 忽略无效词

#### 5.eval

通过搜索进行生成，使用 bleu 评价质量、即 n-gram 的平均重合度