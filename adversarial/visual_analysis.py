import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
import os

from tensorflow.examples.tutorials.mnist import input_data


data = input_data.read_data_sets('../../MNIST_data', one_hot=True)

print("Size of:")
print("- Training-set:\t\t{}".format(len(data.train.labels)))
print("- Test-set:\t\t{}".format(len(data.test.labels)))
print("- Validation-set:\t{}".format(len(data.validation.labels)))

data.test.cls = np.argmax(data.test.labels, axis=1)

img_size = 28
img_size_flat = img_size * img_size
img_shape = (img_size, img_size)
num_channels = 1
num_classes = 10

x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, axis=1)

conv1 = tf.layers.conv2d(inputs=x_image,
                         filters=16, kernel_size=[5, 5],
                         activation=tf.nn.relu)

pool1 = tf.layers.max_pooling2d(inputs=conv1,
                                pool_size=[2, 2], strides=2)

conv2 = tf.layers.conv2d(inputs=pool1,
                         filters=36, kernel_size=[5, 5],
                         activation=tf.nn.relu)

pool2 = tf.layers.max_pooling2d(inputs=conv2,
                                pool_size=[2, 2], strides=2)

pool2_flatten = tf.layers.flatten(pool2)

fc1 = tf.layers.dense(inputs=pool2_flatten, units=128,
                      activation=tf.nn.relu)

logits = tf.layers.dense(inputs=fc1, units=num_classes)

y_pred_cls = tf.argmax(logits, axis=1)

loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=y_true_cls, logits=logits))

optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)

correct_prediction = tf.equal(y_pred_cls,  y_true_cls)

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session()

sess.run(tf.global_variables_initializer())

train_batch_size = 64

num_iterations = 2000

saver = tf.train.Saver()

checkpoint = tf.train.get_checkpoint_state("saved_model")
if checkpoint and checkpoint.model_checkpoint_path:
    saver.restore(sess, checkpoint.model_checkpoint_path)
    print("Successfully loaded:", checkpoint.model_checkpoint_path)
else:
    print("Could not find old network weights")

    for i in range(num_iterations):
        x_batch, y_true_batch = data.train.next_batch(train_batch_size)

        feed_dict_train = {x: x_batch, y_true: y_true_batch}

        sess.run(optimizer, feed_dict=feed_dict_train)

        if (i % 100 == 0) or (i == num_iterations - 1):
            acc = sess.run(accuracy, feed_dict=feed_dict_train)
            msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"
            print(msg.format(i, acc))
    saver.save(sess, 'saved_model/mnist-ana', global_step=num_iterations)


def optimize_image(layer_id=None, feature=0, num_iterations=30):
    tensor = [conv1, pool1, conv2, pool2, fc1, logits][layer_id]
    if layer_id < 4:
        loss = tf.reduce_mean(tensor[:, :, :, feature])
    else:
        loss = tensor[0, feature]

    gradient = tf.gradients(loss, x_image)
    image = 0.1 * np.random.uniform(size=[img_size, img_size, num_channels]) + 0.45

    for i in range(num_iterations):
        feed_dict = {x_image: [image]}
        pred, grad, loss_value = sess.run([y_pred_cls, gradient, loss],
                                          feed_dict=feed_dict)
        if i == num_iterations - 1:
            print('i:', i, 'loss:', loss_value)
        grad = np.array(grad).squeeze()
        step_size = 1 / (grad.std() + 1e-8)
        image += step_size * grad[:, :, np.newaxis]
        image = np.clip(image, 0.0, 1.0)

    return image.squeeze()


def plot_images(images, nrows=2, names='test.png'):
    num = len(images)
    # Create figure with sub-plots.
    fig, axes = plt.subplots(nrows, int(num/nrows))

    # Adjust vertical spacing.
    fig.subplots_adjust(hspace=0.1, wspace=0.1)

    # Use interpolation to smooth pixels?
    smooth = True

    # Interpolation type.
    if smooth:
        interpolation = 'spline16'
    else:
        interpolation = 'nearest'

    # For each entry in the grid.
    for i, ax in enumerate(axes.flat):
        img = images[i]

        # Plot the image.
        ax.imshow(img, interpolation=interpolation, cmap='gray')

        # Remove ticks.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    if not os.path.exists('mnist_layer'):
        os.mkdir('mnist_layer')
    plt.savefig('mnist_layer/{0}'.format(names))


def optimize_images(layer_id, num_feature=6, nrows=2, num_iterations=30, names='test.png'):
    # Initialize the array of images.
    images = []

    # For each feature do the following. Note that the
    # last fully-connected layer only supports numbers
    # between 1 and 1000, while the convolutional layers
    # support numbers between 0 and some other number.
    # So we just use the numbers between 1 and 7.
    for feature in range(0, num_feature):
        print("Optimizing image for feature no.", feature)

        # Find the image that maximizes the given feature
        # for the network layer identified by layer_id (or None).
        image = optimize_image(layer_id=layer_id, feature=feature,
                               num_iterations=num_iterations)

        # Squeeze the dim of the array.
        image = image.squeeze()

        # Append to the list of images.
        images.append(image)

    # Convert to numpy-array so we can index all dimensions easily.
    images = np.array(images)

    # Plot the images.
    plot_images(images=images, nrows=nrows, names=names)


optimize_images(layer_id=0, num_feature=16, nrows=4, names='1_conv1.png')
optimize_images(layer_id=1, num_feature=16, nrows=4, names='1_pool1.png')
optimize_images(layer_id=2, num_feature=36, nrows=6, names='2_conv2.png')
optimize_images(layer_id=3, num_feature=36, nrows=6, names='2_pool2.png')
optimize_images(layer_id=4, num_feature=128, nrows=8, names='3_fc1.png')
optimize_images(layer_id=5, num_feature=10, nrows=2, names='4_logits.png')
