
import numpy as np
import tensorflow as tf
import functions as F

NUM_CLASSES = 10
IMAGE_SIZE = 24
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE


def residual(h, channels, strides):
    h0 = h
    h1 = F.activation(F.batch_normalization(F.conv(h0, channels, strides, bias_term=False)))
    h2 = F.batch_normalization(F.conv(h1, channels, bias_term=False))
    # c.f. http://gitxiv.com/comments/7rffyqcPLirEEsmpX
    print(h0, h2)
    if F.volume(h0) == F.volume(h2):
        h = h2 + h0
    else:
        h3 = F.avg_pool(h0)
        h4 = tf.pad(h3, [[0,0], [0,0], [0,0], [int(channels / 4), int(channels / 4)]])
        h = h2 + h4
    return F.activation(h)

def residual_new(h, channels, strides):
    h0 = h
    h1 = F.conv(F.activation(F.batch_normalization(h0)), channels, strides)
    h2 = F.conv(F.activation(F.batch_normalization(h1)), channels)
    print(h0, h2)
    if F.volume(h0) == F.volume(h2):
        h = h2 + h0
    else:
        h3 = F.avg_pool(h0)
        h4 = tf.pad(h3, [[0,0], [0,0], [0,0], [int(channels / 4), int(channels / 4)]])
        h = h2 + h4
    return h

def inference(images, keep_prob):
    layers = 3
    h = images
    h = F.activation(F.batch_normalization(F.conv(h, 16, bias_term=False)))
    for i in range(layers):
        h = residual(h, channels=16, strides=1)
    for channels in [32, 64]:
        for i in range(layers):
            strides = 2 if i == 0 else 1
            h = residual(h, channels, strides)
    h = tf.reduce_mean(h, reduction_indices=[1, 2]) # Global Average Pooling
    h = F.dense(h, NUM_CLASSES)
    return h

def loss(logits, labels):
    labels = tf.to_int64(labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name='xentropy')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    tf.add_to_collection('losses', cross_entropy_mean)
    return tf.add_n(tf.get_collection('losses'))

def training(loss, learning_rate):
    tf.summary.scalar('loss', loss)
    optimizer = tf.train.AdamOptimizer()

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss)
    return train_op

def evaluating(logits, labels):
    correct = tf.nn.in_top_k(logits, labels, 1)
    correct_sum =  tf.reduce_sum(tf.cast(correct, tf.int32))
    batch_size = labels.shape[0]
    error = 1.0 - tf.cast(correct_sum, tf.float32) / tf.cast(batch_size, tf.float32)
    tf.summary.scalar('error', error)
    return correct_sum
