import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

def conv2D(
    x, 
    out_channel, 
    filter=[3,3], 
    stride=[1,1,1,1], 
    padding='SAME', 
    act=tf.nn.relu, 
    name='conv2d'):
    _, _, _, c = [ i.value for i in x.shape ] # [-1, w, h, c]
    with tf.name_scope(name):
        f = tf.Variable(
            tf.truncated_normal(filter + [c, out_channel], stddev=0.1))
        conv = tf.nn.conv2d(x, f, stride, padding)
        b = tf.Variable(
            tf.constant(0.1, shape=[out_channel]))
        return act(conv + b)

def max_pooling(
    x, 
    ksize=[1,2,2,1], 
    stride=[1,2,2,1],
    padding='SAME', 
    name='max_pooling'):
    with tf.name_scope(name):
        return tf.nn.max_pool(
            x,
            ksize=ksize,
            strides=stride,
            padding=padding)

def dense(x, hidden, act=tf.nn.relu, name='dense'):
    _, h = [ i.value for i in x.shape ] # [-1, h]
    with tf.name_scope(name):
        w = tf.Variable(
            tf.truncated_normal([h, hidden]))
        b = tf.Variable(
            tf.zeros([hidden]))
        return act(tf.matmul(x, w) + b) if act is not None else tf.matmul(x, w) + b

def flatten(x, name='flatten'):
    with tf.name_scope(name):
        return tf.reshape(x, [-1, np.prod([i.value for i in x.shape[1:]])])