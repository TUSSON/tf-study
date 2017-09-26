
import numpy as np
import tensorflow as tf

NUM_CLASSES = 10
IMAGE_SIZE = 24
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

def weight_variable(shape):
    inital = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(inital)

def bias_variable(shape):
    inital = tf.constant(0.1, shape=shape)
    return tf.Variable(inital)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def inference(images, is_training, conv1_units, conv2_units, hidden3_units, hidden4_units):
    x_img = tf.reshape(images, [-1, IMAGE_SIZE, IMAGE_SIZE, 3])
    with tf.name_scope('conv1'):
        weights = weight_variable([5, 5, 3, conv1_units])
        biases = bias_variable([conv1_units])
        conv1 = conv2d(x_img, weights) + biases
        norm1 = tf.nn.relu(tf.layers.batch_normalization(conv1, training=is_training))

    pool1 = max_pool_2x2(norm1)

    with tf.name_scope('conv2'):
        weights = weight_variable([5, 5, conv1_units, conv2_units])
        biases = bias_variable([conv2_units])
        conv2 = conv2d(pool1, weights) + biases
        norm2 = tf.nn.relu(tf.layers.batch_normalization(conv2, training=is_training))

    pool2 = max_pool_2x2(norm2)

    with tf.name_scope('hidden3'):
        dim = 1
        for d in pool2.get_shape()[1:]:
            dim = dim * int(d)
        weights = weight_variable([dim, hidden3_units])
        biases = bias_variable([hidden3_units])
        reshape = tf.reshape(pool2, [-1, dim])
        hidden3 = tf.nn.relu(tf.matmul(reshape, weights) + biases)
    
    with tf.name_scope('hidden4'):
        weights = weight_variable([hidden3_units, hidden4_units])
        biases = bias_variable([hidden4_units])
        hidden4 = tf.nn.relu(tf.matmul(hidden3, weights) + biases)

    with tf.name_scope('sotmax_linear'):
        weights = weight_variable([hidden4_units, NUM_CLASSES])
        biases = bias_variable([NUM_CLASSES])
        logits = tf.matmul(hidden4, weights) + biases

    return logits

def loss(logits, labels):
    labels = tf.to_int64(labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name='xentropy')
    return tf.reduce_mean(cross_entropy, name='xentropy_mean')

def training(loss, learning_rate):
    tf.summary.scalar('loss', loss)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(loss)
    return train_op

def evaluating(logits, labels):
    correct = tf.nn.in_top_k(logits, labels, 1)
    return tf.reduce_sum(tf.cast(correct, tf.int32))
