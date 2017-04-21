#! /usr/bin/env python3

import os

import tensorflow as tf


def wr(g, name):
    name = os.path.join(os.path.dirname(__file__), "{}.pb".format(name))
    open(name, "wb").write(g.as_graph_def().SerializeToString())


g = tf.Graph()
with g.as_default():
    c = tf.constant(123.0)
wr(g, "constant")
