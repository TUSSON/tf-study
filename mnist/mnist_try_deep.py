#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

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

x = tf.placeholder(tf.float32, [None, 784])
x_img = tf.reshape(x, [-1, 28, 28, 1])

W1 = weight_variable([5, 5, 1, 32])
b1 = bias_variable([32])

h1 = tf.nn.relu(conv2d(x_img, W1) + b1)
hp1 = max_pool_2x2(h1)

W2 = weight_variable([5, 5, 32, 64])
b2 = bias_variable([64])
h2 = tf.nn.relu(conv2d(hp1, W2) + b2)
hp2 = max_pool_2x2(h2)

W3 = weight_variable([7*7*64, 1024])
b3 = bias_variable([1024])

hp2_flatten = tf.reshape(hp2, [-1, 7*7*64])
h3 = tf.nn.relu(tf.matmul(hp2_flatten, W3) + b3)

keep_prob = tf.placeholder(tf.float32)
h3_drop = tf.nn.dropout(h3, keep_prob)

W4 = weight_variable([1024, 10])
b4 = bias_variable([10])

y = tf.matmul(h3_drop, W4) + b4

y_ = tf.placeholder(tf.float32, [None, 10])

coss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

#train = tf.train.GradientDescentOptimizer(0.01).minimize(coss)
train = tf.train.AdamOptimizer(1e-4).minimize(coss)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

acc_hist = list()
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    _, cur_acc = sess.run([train, accuracy], feed_dict={x: batch_xs, y_: batch_ys, keep_prob:0.8})
    print(i, cur_acc)
    acc_hist.append(cur_acc)

print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                    y_: mnist.test.labels, keep_prob:1.0}))

#ch1, chp1, ch2, chp2, ch3, cw1, cw2, cw3, cw4  = sess.run([h1, hp1, h2, hp2, h3, W1, W2, W3, W4], feed_dict={x: mnist.test.images,
#                                    y_: mnist.test.labels, keep_prob:1.0})


plt.plot(acc_hist)
plt.show()
