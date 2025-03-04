'''
Please note that the "edges.npy" file is not open-source.
To reproduce the gcn's result, please replicate the process described in the related paper.
'''

import tensorflow as tf
import keras 
import numpy as np
from keras import Sequential, optimizers, losses

channels_num = 64
graph_layer_num = int(1)
graph_convolution_kernel = 5

is_channel_attention = True
class MyChannelAttention(keras.layers.Layer):
    def __init__(self):
        super(MyChannelAttention, self).__init__()
        self.channel_attention = keras.models.Sequential([
            keras.layers.GlobalAvgPool2D(),
            keras.layers.Dense(4, activation='tanh'),
            keras.layers.Dense(channels_num),
        ])

    def build(self, input_shape):
        super(MyChannelAttention, self).build(input_shape)

    def call(self, inputs, **kwargs):
        inputs = tf.transpose(inputs, (0, 1, 3, 2))
        cha_attention = self.channel_attention(inputs)

        cha_attention = tf.reduce_mean(cha_attention, axis=0)

        return cha_attention

    def compute_output_shape(self, input_shape):
        return input_shape


class MyGraphConvolution(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(MyGraphConvolution, self).__init__(**kwargs)
        # 导入邻接矩阵
        adjacency = np.zeros((64, 64))  # solve the adjacency matrix (N*N, eg. 64*64)
        edges = np.load('edges.npy')
        for x, y in edges:
            adjacency[x][y] = 1
            adjacency[y][x] = 1
        adjacency = np.sign(adjacency + np.eye(channels_num))
        adjacency = np.sum(adjacency, axis=0) * np.eye(64) - adjacency
        e_vales, e_vets = np.linalg.eig(adjacency)

        # 计算模型需要的参数
        self.adj = None
        self.e_vales = tf.cast(e_vales, dtype=tf.float32)
        self.e_vets = tf.cast(e_vets, dtype=tf.float32)

        # 计算 图卷积 的卷积核
        graph_kernel = self.add_weight(shape=[graph_convolution_kernel, 1, channels_num])
        graph_kernel = graph_kernel * tf.eye(channels_num)
        graph_kernel = tf.matmul(tf.matmul(self.e_vets, graph_kernel), tf.transpose(self.e_vets, (1, 0)))
        self.graph_kernel = tf.expand_dims(graph_kernel, axis=0)

        # plt.matshow(self.graph_kernel[0,0])

        # 添加 注意力 机制
        self.graph_channel_attention = MyChannelAttention() if is_channel_attention else []

    def build(self, input_shape, **kwargs):
        super(MyGraphConvolution, self).build(input_shape)  # 一定要在最后调用它

    # x：batch * k * channels * times, 16 * 1||5 * 64 * 128
    def call(self, x, **kwargs):
        adj = self.graph_kernel

        # 通道注意力网络
        if is_channel_attention:
            cha_attention = self.graph_channel_attention(x)
            adj = cha_attention * adj

            # 卷积过程
        x = tf.matmul(adj, x)
        x = keras.layers.Activation('relu')(x)

        return x

    @staticmethod
    def compute_output_shape(input_shape):
        return input_shape


def create_GCNmodel(sample_len, lr):
    # set the model
    model = Sequential()

    # the input data preprocess
    model.add(keras.layers.Permute((2, 1), input_shape=(sample_len, channels_num)))
    model.add(keras.layers.Reshape((1, channels_num, sample_len)))
    model.add(keras.layers.BatchNormalization(axis=1))

    # convolution module
    for k in range(graph_layer_num):
        model.add(MyGraphConvolution())
        model.add(keras.layers.BatchNormalization(axis=1))

    # 全连接分类器
    model.add(keras.layers.Permute((1, 3, 2)))
    model.add(keras.layers.AvgPool2D((1, sample_len)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Dense(8, activation='tanh'))
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Dense(2, activation='softmax'))

    # set the optimizers
    model.compile(
        optimizer=optimizers.Adam(lr),
        loss=losses.CategoricalCrossentropy(),
        metrics=['accuracy'],
    )

    return model
