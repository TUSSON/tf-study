import numpy as np
import tensorflow as tf

from datetime import datetime
import argparse
import os
import sys
import time
import math

FLAGS = None

import cifar10
from cifar10_input import *

def eval_once(saver, summary_writer, top_k_op, summary_op, data_sets):
  """Run Eval once.

  Args:
    saver: Saver.
    summary_writer: Summary writer.
    top_k_op: Top K op.
    summary_op: Summary op.
  """
  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint
      saver.restore(sess, ckpt.model_checkpoint_path)
      # Assuming model_checkpoint_path looks something like:
      #   /my-favorite-path/cifar10_train/model.ckpt-0,
      # extract global_step from it.
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
      print('No checkpoint file found')
      return

    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))

      num_iter = int(math.ceil(data_sets.num_examples / FLAGS.batch_size))
      true_count = 0  # Counts the number of correct predictions.
      total_sample_count = num_iter * FLAGS.batch_size
      step = 0
      while step < num_iter and not coord.should_stop():
        predictions = sess.run([top_k_op])
        true_count += np.sum(predictions)
        step += 1

      # Compute precision @ 1.
      precision = true_count / total_sample_count
      print('{0}: precision @ 1 = {1:0.03%}'.format(datetime.now(), precision))

      summary = tf.Summary()
      summary.ParseFromString(sess.run(summary_op))
      summary.value.add(tag='Precision @ 1', simple_value=precision)
      summary_writer.add_summary(summary, global_step)
    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)


def evaluate():
  """Eval CIFAR-10 for a number of steps."""
  with tf.Graph().as_default():
    # Get images and labels for CIFAR-10.
    with tf.name_scope('input'):
        test_data_sets  = DataSet(test_images, test_labels, FLAGS.batch_size, False)
    images, labels = test_data_sets.images, test_data_sets.labels

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = cifar10.inference(images, 1.0)

    # Calculate predictions.
    top_k_op = tf.nn.in_top_k(logits, labels, 1)

    saver = tf.train.Saver()

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()

    graph_def = tf.get_default_graph().as_graph_def()
    summary_writer = tf.summary.FileWriter(FLAGS.eval_dir,
                                            graph_def=graph_def)

    while True:
      eval_once(saver, summary_writer, top_k_op, summary_op, test_data_sets)
      time.sleep(FLAGS.eval_interval_secs)

def main(_):
    evaluate()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--eval_interval_secs',
        type=int,
        default=60*5,
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=128,
        help='Batch size.  Must divide evenly into the dataset sizes.'
    )
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                             'tensorflow/cifar10/logs/error'),
        help='Directory to put the log data.'
    )
    parser.add_argument(
        '--eval_dir',
        type=str,
        default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                             'tensorflow/cifar10/logs/eval'),
        help='Directory to put the log data.'
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed) 
