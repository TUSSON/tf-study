import numpy as np
import tensorflow as tf

import argparse
import os
import sys
import time

from tensorflow.contrib.keras.api.keras.datasets.cifar10 import load_data
import cifar10

# Basic model parameters as external flags.
FLAGS = None

class DataSet(object):
    def __init__(self, images, labels, is_train_data):
        self._num_examples = images.shape[0]
        self._images = tf.constant(images)
        self._is_train_data = is_train_data
        self._labels = tf.constant(labels)
        self._index_in_epoch = 0
        self._distorted_data()

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    def _generate_image_and_label_batch(self, image, label):
        """Construct a queued batch of images and labels.

        Args:
            image: 3-D Tensor of [IMAGE_SIZE, IMAGE_SIZE, 3] of type.float32.
            label: 1-D Tensor of type.int32
            in the queue that provides of batches of examples.

        Returns:
            images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
            labels: Labels. 1D tensor of [batch_size] size.
        """

        min_queue_examples = int(self.num_examples * 0.4)
        # Create a queue that shuffles the examples, and then
        # read 'FLAGS.batch_size' images + labels from the example queue.
        num_preprocess_threads = 2
        images, label_batch = tf.train.shuffle_batch(
            [image, label],
            batch_size=FLAGS.batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * FLAGS.batch_size,
            min_after_dequeue=min_queue_examples)

        # Display the training images in the visualizer.
        tf.summary.image('images', images)
        self._images = images
        self._labels = tf.reshape(label_batch, [FLAGS.batch_size])

    def _distorted_data(self):
        image, label = tf.train.slice_input_producer([self._images, self._labels])

        image = tf.cast(image, tf.float32)
        w = h = cifar10.IMAGE_SIZE
        image = tf.image.resize_image_with_crop_or_pad(image, h, w)
        if self._is_train_data:
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_brightness(image, max_delta=63)
            image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
        image = tf.image.per_image_standardization(image)
        self._generate_image_and_label_batch(image, label)

def placeholder_inputs(batch_size):
    images_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
                                                           cifar10.IMAGE_SIZE, cifar10.IMAGE_SIZE, 3))
    labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
    is_training_placeholder = tf.placeholder(tf.bool)
    return images_placeholder, labels_placeholder, is_training_placeholder

def do_eval(sess,
            eval_correct,
            images_placeholder,
            labels_placeholder,
            is_training_placeholder,
            data_set):
    true_count = 0
    steps_per_epoch = data_set.num_examples // FLAGS.batch_size
    num_examples = steps_per_epoch * FLAGS.batch_size
    for step in range(steps_per_epoch):
        image, label = sess.run([data_set.images, data_set.labels])
        feed_dict = {
            images_placeholder: image,
            labels_placeholder: label,
            is_training_placeholder: False
        }
        true_count += sess.run(eval_correct, feed_dict=feed_dict)
    precision = float(true_count) / num_examples
    print('  Num examples: {0}  Num correct: {1}  Precision @ 1: {2:0.04%}'.format(
          num_examples, true_count, precision))
        
def run_training():
    train_data, test_data = load_data()
    with tf.Graph().as_default():
        with tf.name_scope('input'):
            train_data_sets = DataSet(train_data[0], train_data[1], True)
            test_data_sets  = DataSet(test_data[0], test_data[1], False)
        images_placeholder, labels_placeholder, is_training_placeholder = placeholder_inputs(FLAGS.batch_size)

        logits = cifar10.inference(images_placeholder,
                                   is_training_placeholder,
                                   FLAGS.conv1_units,
                                   FLAGS.conv2_units,
                                   FLAGS.hidden3_units,
                                   FLAGS.hidden4_units)

        loss = cifar10.loss(logits, labels_placeholder)

        train_op = cifar10.training(loss, FLAGS.learning_rate)

        eval_correct = cifar10.evaluating(logits, labels_placeholder)

        summary = tf.summary.merge_all()

        init = tf.global_variables_initializer()

        saver = tf.train.Saver()

        sess = tf.Session()

        summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

        sess.run(init)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for step in range(FLAGS.max_steps):
            start_time = time.time()

            image, label = sess.run([train_data_sets.images, train_data_sets.labels])
            feed_dict = {
                images_placeholder: image,
                labels_placeholder: label,
                is_training_placeholder: True
            }

            _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)

            duration = time.time() - start_time

            if step % 10 == 0:
                print('Step {0}: loss = {1:.2} {2:.3} sec'.format(step, loss_value, duration))
                summary_str = sess.run(summary, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()

            if (step + 1) % 200 == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_file = os.path.join(FLAGS.log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_file, global_step=step)
                if (step + 1)%1000 == 0:
                    print('Training Data Eval:')
                    do_eval(sess, eval_correct,
                            images_placeholder, labels_placeholder,
                            is_training_placeholder,train_data_sets)
                print('Test Data Eval:')
                do_eval(sess, eval_correct,
                        images_placeholder, labels_placeholder,
                        is_training_placeholder,test_data_sets)

def main(_):
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    run_training()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.0001,
        help='Initial learning rate.'
    )
    parser.add_argument(
        '--max_steps',
        type=int,
        default=20000,
        help='Number of steps to run trainer.'
    )
    parser.add_argument(
        '--conv1_units',
        type=int,
        default=64,
        help='Number of units in conv layer 1.'
    )
    parser.add_argument(
        '--conv2_units',
        type=int,
        default=64,
        help='Number of units in conv layer 2.'
    )
    parser.add_argument(
        '--hidden3_units',
        type=int,
        default=384,
        help='Number of units in hidden layer 3.'
    )
    parser.add_argument(
        '--hidden4_units',
        type=int,
        default=192,
        help='Number of units in hidden layer 3.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=100,
        help='Batch size.  Must divide evenly into the dataset sizes.'
    )
    parser.add_argument(
        '--keep_prob',
        type=float,
        default=0.8
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                             'tensorflow/cifar10/logs/fully_connected_feed'),
        help='Directory to put the log data.'
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed) 
