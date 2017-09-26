import tensorflow as tf
import numpy as np

# Model parameters
W = tf.Variable([0.3], dtype=tf.float32)
b = tf.Variable([-0.3], dtype=tf.float32)
# Model input and output
x = tf.placeholder(tf.float32)
linear_model = W * x + b
y = tf.placeholder(tf.float32)

# loss
loss = tf.reduce_sum(tf.square(linear_model - y))
# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# training data
x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]

# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
for i in range(1000):
    sess.run(train, {x: x_train, y: y_train})
    _, curr_W, curr_b, curr_loss = sess.run([train, W, b, loss], {x: x_train, y: y_train})
    print(i, 'W:', curr_W, 'b:', curr_b, 'loss:', curr_loss)
