#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.truncated_normal([784, 100], stddev=0.1))
b = tf.Variable(tf.zeros([100]))

W2 = tf.Variable(tf.truncated_normal([100, 10], stddev=0.1))
b2 = tf.Variable(tf.zeros([10]))

y0 = tf.nn.relu(tf.matmul(x, W) + b)

keep_prob = tf.placeholder(tf.float32)
h_drop = tf.nn.dropout(y0, keep_prob)

y = tf.matmul(h_drop, W2) + b2

y_ = tf.placeholder(tf.float32, [None, 10])

coss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

#train = tf.train.GradientDescentOptimizer(0.01).minimize(coss)
train = tf.train.AdamOptimizer(1e-4).minimize(coss)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

acc_hist = list()
for i in range(20000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    _, cur_acc = sess.run([train, accuracy], feed_dict={x: batch_xs, y_: batch_ys, keep_prob:0.5})
    acc_hist.append(cur_acc)

print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                    y_: mnist.test.labels, keep_prob:1.0}))

plt.plot(acc_hist)
plt.show()
