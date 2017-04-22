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
