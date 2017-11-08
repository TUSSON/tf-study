import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math

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


def plot_images(images, cls_true, cls_pred=None, noise=0.0):
    assert len(images) == len(cls_true) == 9

    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Get the i'th image and reshape the array.
        image = images[i].reshape(img_shape)

        # Add the adversarial noise to the image.
        image += noise

        # Ensure the noisy pixel-values are between 0 and 1.
        image = np.clip(image, 0.0, 1.0)

        # Plot image.
        ax.imshow(image,
                  cmap='binary', interpolation='nearest')

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, axis=1)

noise_limit = 0.35
noise_l2_wieght = 0.02

ADVERSARY_VARIABLES = 'adversary_variables'
collections = [tf.GraphKeys.GLOBAL_VARIABLES, ADVERSARY_VARIABLES]

x_noise = tf.Variable(tf.zeros([img_size, img_size, num_channels]),
                      name='x_noise', trainable=False,
                      collections=collections)
x_noise_clip = tf.assign(x_noise, tf.clip_by_value(x_noise,
                                                   -noise_limit,
                                                   noise_limit))

x_noisy_image = x_image + x_noise
x_noisy_image = tf.clip_by_value(x_noisy_image, 0.0, 1.0)

conv1 = tf.layers.conv2d(inputs=x_noisy_image,
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

print([var.name for var in tf.trainable_variables()])

adversary_variables = tf.get_collection(ADVERSARY_VARIABLES)

print([var.name for var in adversary_variables])

l2_loss_noise = noise_l2_wieght * tf.nn.l2_loss(x_noise)

loss_adversary = loss + l2_loss_noise

optimizer_adversary = tf.train.AdamOptimizer(learning_rate=1e-2).minimize(
    loss_adversary, var_list=adversary_variables)

correct_prediction = tf.equal(y_pred_cls,  y_true_cls)

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session()

sess.run(tf.global_variables_initializer())


def init_noise():
    sess.run(tf.variables_initializer([x_noise]))


init_noise()

train_batch_size = 64


def optimize(num_iterations, adversary_target_cls=None):
    start_time = time.time()

    for i in range(num_iterations):
        x_batch, y_true_batch = data.train.next_batch(train_batch_size)
        if adversary_target_cls is not None:
            y_true_batch = np.zeros_like(y_true_batch)
            y_true_batch[:, adversary_target_cls] = 1.0

        feed_dict_train = {x: x_batch, y_true: y_true_batch}

        if adversary_target_cls is None:
            sess.run(optimizer, feed_dict=feed_dict_train)
        else:
            sess.run(optimizer_adversary, feed_dict=feed_dict_train)
            sess.run(x_noise_clip)

        if (i % 100 == 0) or (i == num_iterations - 1):
            acc = sess.run(accuracy, feed_dict=feed_dict_train)
            msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"
            print(msg.format(i, acc))

    end_time = time.time()
    time_dif = end_time - start_time
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))


def get_noise():
    noise = sess.run(x_noise)
    return np.squeeze(noise)


def plot_noise():
    noise = get_noise()

    print("Noise:")
    print("- Min:", noise.min())
    print("- Max:", noise.max())
    print("- Std:", noise.std())

    plt.imshow(noise, interpolation='nearest', cmap='seismic',
               vmin=-1.0, vmax=1.0)


def plot_example_errors(cls_pred, correct):
    incorrect = (correct == False)
    images = data.test.images[incorrect]
    cls_pred = cls_pred[incorrect]
    cls_true = data.test.cls[incorrect]
    noise = get_noise()
    plot_images(images=images[0:9],
                cls_true=cls_true[0:9],
                cls_pred=cls_pred[0:9],
                noise=noise)


def plot_confusion_matrix(cls_pred):
    cls_true = data.test.cls
    cm = confusion_matrix(y_true=cls_true,
                          y_pred=cls_pred)

    print(cm)


test_batch_size = 256


def print_test_accuracy(show_example_errors=False,
                        show_confusion_matrix=False):
    num_test = len(data.test.images)
    cls_pred = np.zeros(shape=num_test, dtype=np.int)

    i = 0
    while i < num_test:
        j = min(i + test_batch_size, num_test)
        images = data.test.images[i:j, :]
        labels = data.test.labels[i:j, :]

        feed_dict = {x: images, y_true: labels}
        cls_pred[i:j] = sess.run(y_pred_cls, feed_dict=feed_dict)
        i = j

    cls_true = data.test.cls
    correct = (cls_true == cls_pred)
    correct_sum = correct.sum()
    acc = float(correct_sum) / num_test
    msg = 'Accuracy on Test-Set: {0:.1%} ({1} / {2})'
    print(msg.format(acc, correct_sum, num_test))
    if show_example_errors:
        print("Example errors:")
        plot_example_errors(cls_pred=cls_pred, correct=correct)

    # Plot the confusion matrix, if desired.
    if show_confusion_matrix:
        print("Confusion Matrix:")
        plot_confusion_matrix(cls_pred=cls_pred)


optimize(num_iterations=1000)

print_test_accuracy(show_example_errors=True)

init_noise()

def find_all_noise(num_iterations=1000):
    all_noise = []

    for i in range[num_classes]:
        print('Finding adversarial noise for target-class:', i)

        init_noise()

        optimize(num_iterations=num_iterations, adversary_target_cls=i)

        noise = get_noise()

        all_noise.append(noise)

        print()

    return all_noise


# all_noise = find_all_noise(num_iterations=300)


def plot_all_noise(all_noise):
    fig, axes = plt.subplots(2, 5)
    fig.subplots_adjust(hspace=0.2, wspace=0.1)

    # For each sub-plot.
    for i, ax in enumerate(axes.flat):
        # Get the adversarial noise for the i'th target-class.
        noise = all_noise[i]

        # Plot the noise.
        ax.imshow(noise,
                  cmap='seismic', interpolation='nearest',
                  vmin=-1.0, vmax=1.0)

        # Show the classes as the label on the x-axis.
        ax.set_xlabel(i)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


def make_immune(target_cls, num_iterations_adversary=500,
                num_iterations_immune=200):
    print("Target-class:", target_cls)
    print("Finding adversarial noise ...")

    optimize(num_iterations=num_iterations_adversary,
             adversary_target_cls=target_cls)

    print()

    print_test_accuracy(show_example_errors=False,
                        show_confusion_matrix=False)

    print()

    print("Making the neural network immune to the noise ...")

    optimize(num_iterations=num_iterations_immune)

    print()

    print_test_accuracy(show_example_errors=False,
                        show_confusion_matrix=False)


for i in range(10):
    make_immune(target_cls=i)
    print()
    make_immune(target_cls=i)
    print()


print_test_accuracy(show_example_errors=False,
                    show_confusion_matrix=False)


init_noise()

print_test_accuracy(show_example_errors=True,
                    show_confusion_matrix=True)
