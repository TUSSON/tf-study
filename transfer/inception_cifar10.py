import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.contrib.keras.api.keras.datasets.cifar10 import load_data

import argparse
import os
import sys
from download import maybe_download_and_extract
from cache import cache

inception_url = "http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz"
inception_dir = 'inception/'
inception_graph_def_path = "classify_image_graph_def.pb"

(train_images, train_labels), (test_images, test_labels) = load_data()
train_labels = train_labels.ravel()
test_labels = test_labels.ravel()
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

def plot_images(images, labels, pred_labels=None, nrows=5, nclos=5):
    assert len(images) == len(labels)
    assert nrows > 0 or nclos > 0

    img_num = len(images)
    if nrows <= 0:
        nrows = img_num / nclos
    elif nclos <= 0:
        nclos = img_num / nrows

    f, axs = plt.subplots(nrows, nclos)
    f.subplots_adjust(left=0, right=0.95, top=1, bottom=0.1, hspace=1.0, wspace=1.0)

    for i, ax in enumerate(axs.flat):
        ax.set_xticks([])
        ax.set_yticks([])
        if i >= img_num:
            break
        ax.imshow(images[i])
        label_name = class_names[labels[i]]
        if pred_labels is not None:
            pred_name = class_names[pred_labels[i]]
            xlabel = 'T:{0}\nP:{1}'.format(label_name, pred_name)
        else:
            xlabel = 'T:{0}'.format(label_name)

        ax.set_xlabel(xlabel)

    plt.show()


def transfer_values_cache(cache_path, sess, input_layer, layer, images):
    def transfer_images():
        transfer_values = []
        for i, img in enumerate(images):
            print('transfer image:', i)
            value = sess.run(layer, feed_dict={input_layer: img})
            transfer_values.append(value.reshape(-1))
        return np.array(transfer_values)
    return cache(cache_path, transfer_images)


def run_training():
    g = tf.Graph()
    with g.as_default():
        path = os.path.join(inception_dir, inception_graph_def_path)
        with tf.gfile.FastGFile(path, 'rb') as file:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(file.read())
            tf.import_graph_def(graph_def, name='')

    input_image = g.get_tensor_by_name('DecodeJpeg:0')

    transfer_layer = g.get_tensor_by_name('pool_3:0')
    transfer_len = transfer_layer.get_shape()[-1]
    sess = tf.Session(graph=g)

    transfer_train_values = transfer_values_cache(
        'inception_cifar10_train.pkl',
        sess, input_image, transfer_layer,
        train_images)

    transfer_test_values = transfer_values_cache(
        'inception_cifar10_test.pkl',
        sess, input_image, transfer_layer, test_images)

    x = tf.placeholder(tf.float32, shape=(None, transfer_len))
    y_class = tf.placeholder(tf.float32, shape=(None))

    y_pred = tf.layers.dense(inputs=x, units=10)
    y_pred_class = tf.argmax(y_pred, axis=1)

    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=tf.to_int64(y_class),
        logits=y_pred))

    optimizer = tf.train.AdamOptimizer(learning_rate=1e-5)
    train_op = optimizer.minimize(loss)

    correct_prediction = tf.equal(y_pred_class, tf.to_int64(y_class))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for step in range(50000):
        batch_idx = np.random.choice(len(transfer_train_values), 128)
        x_batch = transfer_train_values[batch_idx]
        y_batch = train_labels[batch_idx]
        cur_accuracy, cur_loss, _ = sess.run((accuracy, loss, train_op),
                                             feed_dict={x: x_batch,
                                                        y_class: y_batch})
        if step % 100 == 0:
            print('step:', step, 'loss:', cur_loss, 'accuracy:{:0.03%}'.format(cur_accuracy))

    test_accuracy, test_pred = sess.run((accuracy, y_pred_class),
                                        feed_dict={x: transfer_test_values,
                                        y_class: test_labels})
    print('test accuracy:{:0.03%}'.format(test_accuracy))

    error_idx = test_pred != test_labels
    err_images = test_images[error_idx]
    err_label  = test_labels[error_idx]
    err_pred_labels = test_pred[error_idx]
    plot_images(err_images, err_label, err_pred_labels)

    sess.close()


def main(_):
    maybe_download_and_extract(inception_url, inception_dir)
    run_training()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
