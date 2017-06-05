from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None

import os
from contextlib import contextmanager

@contextmanager
def gen(name):
    name = os.path.join(os.getcwd(), "{}.pbtxt".format(name))
    g = tf.Graph()
    with g.as_default():
        yield
    #tf.train.export_meta_graph(graph_def=g.as_graph_def(), filename=name, as_text=True)
    tf.train.write_graph(g, '.', name, as_text=True)

def main(_):
  with gen("mnist/mnist_simple"):
      # Import data
      mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

      # Create the model
      x = tf.placeholder(tf.float32, [None, 784], name="x")
      W = tf.Variable(tf.zeros([784, 10]), name="W")
      b = tf.Variable(tf.zeros([10]), name="b")
      y = tf.matmul(x, W) + b

      # Define loss and optimizer
      y_ = tf.placeholder(tf.float32, [None, 10])

      # The raw formulation of cross-entropy,
      #
      #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
      #                                 reduction_indices=[1]))
      #
      # can be numerically unstable.
      #
      # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
      # outputs of 'y', and then average across the batch.
      cross_entropy = tf.reduce_mean(
          tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
      train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

      sess = tf.InteractiveSession()
      tf.global_variables_initializer().run()
      # Train
      for _ in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

      # Test trained model
      correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
      print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                          y_: mnist.test.labels}))
      tf.train.Saver().save(sess, 'mnist/mnist_simple')

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)