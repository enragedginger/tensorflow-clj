#! /usr/bin/env python3

import os
from contextlib import contextmanager

import tensorflow as tf


@contextmanager
def gen(name):
    name = os.path.join(os.path.dirname(__file__), "{}.pb".format(name))
    with open(name, "wb") as out:
        g = tf.Graph()
        with g.as_default():
            yield
        out.write(g.as_graph_def().SerializeToString())


with gen("constant"):
    tf.constant(123.0)

with gen("addconst"):
    tf.constant(3.0) * tf.placeholder(tf.float32)

with gen("mulbymat"):
    tf.placeholder(tf.float32) * tf.constant([[1., 2.], [3., 4.]])

with gen("mul2vars"):
    tf.placeholder(tf.float32, name="a") * \
        tf.placeholder(tf.float32, name="b")

with gen("linreg"):
    W = tf.Variable([.3], tf.float32, name="W")
    b = tf.Variable([-.3], tf.float32, name="b")
    x = tf.placeholder(tf.float32, name="x")
    linear_model = W * x + b
    tf.identity(linear_model, name="linear_model")

    y = tf.placeholder(tf.float32, name="y")
    squared_deltas = tf.square(linear_model - y, name="squared_deltas")
    loss = tf.reduce_sum(squared_deltas, name="loss")

    fixW = tf.assign(W, [-1.], name="fixW")
    fixb = tf.assign(b, [1.], name="fixb")

    init = tf.global_variables_initializer()
    #tf.identity(init, name="init")
