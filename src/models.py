"""
Models for supervised meta-learning.
"""

from functools import partial

import numpy as np
import tensorflow as tf

DEFAULT_OPTIMIZER = partial(tf.train.AdamOptimizer, beta1=0)

# pylint: disable=R0903
class OmniglotModel:
    """
    A model for Omniglot classification.
    """
    def __init__(self, num_classes, num_filters, optimizer=DEFAULT_OPTIMIZER, **optim_kwargs):
        self.input_ph = tf.placeholder(tf.float32, shape=(None, 28, 28))
        out = tf.reshape(self.input_ph, (-1, 28, 28, 1))
        for _ in range(4):
            out = tf.layers.conv2d(out, num_filters, 3, strides=2, padding='same')
            out = tf.layers.batch_normalization(out, training=True)
            out = tf.nn.relu(out)
        out = tf.reshape(out, (-1, int(np.prod(out.get_shape()[1:]))))
        self.logits = tf.layers.dense(out, num_classes)
        self.label_ph = tf.placeholder(tf.int32, shape=(None,))
        self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label_ph,
                                                                   logits=self.logits)
        self.predictions = tf.argmax(self.logits, axis=-1)
        self.minimize_op = optimizer(**optim_kwargs).minimize(self.loss)

# pylint: disable=R0903
class MiniImageNetModel:
    """
    A model for Mini-ImageNet classification.
    """
    def __init__(self, num_classes, num_filters, optimizer=DEFAULT_OPTIMIZER, **optim_kwargs):
        self.input_ph = tf.placeholder(tf.float32, shape=(None, 84, 84, 3))
        out = self.input_ph
        for _ in range(4):
            out = tf.layers.conv2d(out, num_filters, 3, padding='same')
            out = tf.layers.batch_normalization(out, training=True)
            out = tf.layers.max_pooling2d(out, 2, 2, padding='same')
            out = tf.nn.relu(out)
        out = tf.reshape(out, (-1, int(np.prod(out.get_shape()[1:]))))
        self.logits = tf.layers.dense(out, num_classes)
        self.label_ph = tf.placeholder(tf.int32, shape=(None,))
        self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label_ph,
                                                                   logits=self.logits)
        self.predictions = tf.argmax(self.logits, axis=-1)
        self.minimize_op = optimizer(**optim_kwargs).minimize(self.loss)

class ResNet_Block:
    """
    Be used to construct a resnet block
    """
    def __init__(self,downsample=False):

        self._downsample = downsample
        if downsample:
            self.stride = 2
        else:
            self.stride = 1

        self.kernel_size = 3

    def forward(self,input, out_filter, padding_opt='same'):
        # determine the indentified block
        identi_part = input
        # the first conv in res-block
        out = tf.layers.conv2d(input, out_filter, self.kernel_size, self.stride, padding_opt)
        out = tf.layers.batch_normalization(out, training=True)
        out = tf.nn.relu(out)

        # the 2nd conv layer in res-block
        out = tf.layers.conv2d(out, out_filter, self.kernel_size, strides=1, padding=padding_opt)
        out = tf.layers.batch_normalization(out, training=True)
        if self._downsample:
            out = tf.nn.relu(out)
        else:
            out = tf.nn.relu(out + identi_part)
        return out

class ResNet18Model:
    """
       A ResNet model for miniimagenet classification
    """

    def __init__(self,num_classes, optimizer=DEFAULT_OPTIMIZER, **optim_kwargs):
        self.input_ph = tf.placeholder(tf.float32,shape=(None, 224, 224, 3))
        out = self.input_ph

        # the convnet layer
        out = tf.layers.conv2d(out, 64, 7, strides=2, padding='same')
        out = tf.layers.batch_normalization(out, training=True)
        out = tf.layers.max_pooling2d(out, 3, 2, padding='same')
        out = tf.nn.relu(out)

        # the 1st res-block
        for _ in range(2):
            res_block_1 = ResNet_Block(downsample=False)
            out = res_block_1.forward(out, 64)

        # the 2nd res-block
        res_block_2 = ResNet_Block(downsample=True)
        out = res_block_2.forward(out, 128)
        res_block_2 = ResNet_Block(downsample=False)
        out = res_block_2.forward(out, 128)

        # the 3nd res-block
        res_block_3 = ResNet_Block(downsample=True)
        out = res_block_3.forward(out, 256)
        res_block_3 = ResNet_Block(downsample=False)
        out = res_block_3.forward(out, 256)

        # the 4th res-block
        res_block_4 = ResNet_Block(downsample=True)
        out = res_block_4.forward(out, 512)
        res_block_4 = ResNet_Block(downsample=False)
        out = res_block_4.forward(out, 512)

        # the average_pool layer
        out = tf.layers.average_pooling2d(out, 7, strides=1, padding='same')

        # reshape the size of the feature map
        out = tf.reshape(out, (-1, int(np.prod(out.get_shape()[1:]))))

        # the fully connected layer
        self.logits = tf.layers.dense(out,num_classes)

        self.label_ph = tf.placeholder(tf.int32,shape=(None,))
        # compute the cross_entropy loss
        self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label_ph,
                                                                   logits=self.logits)
        self.predictions = tf.argmax(self.logits,axis=-1)
        self.minimize_op = optimizer(**optim_kwargs).minimize(self.loss)