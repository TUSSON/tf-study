import numpy as np
import tensorflow as tf

import argparse
import os
import sys
import time

FLAGS = None

import cifar10
from cifar10_input import *

# Basic model parameters as external flags.

def placeholder_inputs(batch_size):
    images_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
                                                           cifar10.IMAGE_SIZE, cifar10.IMAGE_SIZE, 3))
    labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
    is_training_placeholder = tf.placeholder(tf.float32)
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
            is_training_placeholder: 1.0
        }
        true_count += sess.run(eval_correct, feed_dict=feed_dict)
    precision = float(true_count) / num_examples
    print('  Num examples: {0}  Num correct: {1}  Precision @ 1: {2:0.04%}'.format(
          num_examples, true_count, precision))
    return precision
        
def run_training():
    with tf.Graph().as_default():
        with tf.name_scope('input'):
            train_data_sets = DataSet(train_images, train_labels, FLAGS.batch_size, True)
            test_data_sets  = DataSet(test_images, test_labels, FLAGS.batch_size, False)
        images_placeholder, labels_placeholder, is_training_placeholder = placeholder_inputs(FLAGS.batch_size)

        logits = cifar10.inference(images_placeholder,
                                   is_training_placeholder)

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

        try:
            start_time = time.time()
            correct_counts = 0
            for step in range(FLAGS.max_steps):
                if coord.should_stop():
                    break

                image, label = sess.run([train_data_sets.images, train_data_sets.labels])
                feed_dict = {
                    images_placeholder: image,
                    labels_placeholder: label,
                    is_training_placeholder: 0.5
                }

                _, loss_value, correct_count = sess.run([train_op, loss, eval_correct], feed_dict=feed_dict)
                correct_counts += correct_count

                if step % 10 == 0:
                    duration = time.time() - start_time
                    start_time = time.time()
                    precision = float(correct_counts) / (FLAGS.batch_size * 10.0)
                    correct_counts = 0
                    print('Step {0}: loss = {1:.2} {2:.3} sec precision: {3:0.04%}'.format(step, loss_value, duration, precision))
                    summary_str = sess.run(summary, feed_dict=feed_dict)
                    summary_writer.add_summary(summary_str, step)
                    summary_writer.flush()

                if (step + 1) % 200 == 0 or (step + 1) == FLAGS.max_steps:
                    checkpoint_file = os.path.join(FLAGS.log_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_file, global_step=step)
                    #if (step + 1)%1000 == 0:
                    #    print('Training Data Eval:')
                    #    train_precision = do_eval(sess, eval_correct,
                    #            images_placeholder, labels_placeholder,
                    #            is_training_placeholder,train_data_sets)
                    #print('Test Data Eval:')
                    #do_eval(sess, eval_correct,
                    #        images_placeholder, labels_placeholder,
                    #        is_training_placeholder,test_data_sets)
        except Exception as e:
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads)

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
        default=40000,
        help='Number of steps to run trainer.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=128,
        help='Batch size.  Must divide evenly into the dataset sizes.'
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
