import tensorflow as tf
import numpy as np

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

w = tf.get_variable("w", shape=[3, 1])

f = tf.stack([tf.square(x), x, tf.ones_like(x)], 1)
yhat = tf.squeeze(tf.matmul(f, w), 1)

loss = tf.nn.l2_loss(yhat - y) + 0.1*tf.nn.l2_loss(w)
optimizer = tf.train.AdamOptimizer(1.0)
train = optimizer.minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1000):
    x_train  = np.random.uniform(-1.0, 1.0, 100)
    y_train = 5.0 * x_train * x_train + 3.0
    _, cur_w, loss_val = sess.run([train, w, loss], {x:x_train, y: y_train})
    print(i, cur_w[0], cur_w[1], cur_w[2], loss_val)
