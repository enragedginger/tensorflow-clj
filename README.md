# tensorflow-clj

[![Build Status](https://travis-ci.org/enragedginger/tensorflow-clj.svg?branch=master)](https://travis-ci.org/enragedginger/tensorflow-clj)

This project is under active development. Expect the APIs to change frequently for the next several months.

## Summary
Clojure API for building and running computations on Google's TensorFlow framework.

## Goal(s)
* Create a mechanism for solving hard problems through machine learning that doesn't require a deep understanding of machine learning and utilizes idiomatic Clojure practices, guidelines, and ideas.

## Rationale (because it wouldn't be a Clojure library without this section)
There exist numerous hard problems that can be solved through machine learning. However, most machine learning frameworks
and libraries require thorough knowledge of the complex foundational topics of the field. In some cases, these libraries
assume the user has sufficient breadth of understanding to know which particular algorithm / approach should be selected
at a given decision point in their venture.

Furthermore, most machine learning libraries are architected in such a way that code re-use across projects for data
scientists is either a copy-paste extravaganza or simply impossible. Oftentimes, this is the result of object oriented
or procedural design principles and practices.

Therefore, `tensorflow-clj` focuses on empowering machine learning plebians to solve hard problems
through the utilization of high-level constructs and automated tooling. However, this library will also allow
machine learning gurus to compose basic machine learning building blocks into constructs they require to solve hard
problems. This will be achieved by taking a data-first, functional approach to automate the building of Tensorflow
graphs.

## Milestones
* Load, run, update, and save Tensorflow graphs which are already available in the pre-defined TF Protobuf format (done)
* Build a representation for any arbitrary Tensorflow operation nodes (done)
* Convert collections of Clojure TF op nodes to TF Protobuf format (done)
(At this point, we can build, load, run, update, and save any arbitrary TF computation graph)
* Mimic convenience functionality present only in Python client (looking at you, GradientDescent)
* Generate `clojure.spec` node schemas based on `op` definitions found in TF Protobuf exports
* Create `clojure.spec` schemas for governing entire collections of nodes
* Mimic convenience functionality present in Keras and similar libraries for building neural nets
* Add functionality for building ML / NN graphs for the plebians


## How to Get Involved
* You can find and chat with us on **#tensorflow** @ Clojurians Slack. [Get an invite here.](http://clojurians.net/)
* If you find bugs / have feature requests, feel free to make an issue here on GitHub. For bugs, please provide sample
code where possible for reproducing the issue. Also, be sure to let us know what environment (OS, Java version, CLJ version,
etc.) you're using.

## Usage

This library is available via Clojars: `[tensorflow-clj "0.1"]`

This project should run out-of-the-box using Leiningen. It's developed locally on MacOS and also built on Travis CI Linux.

More usage instructions coming soon!

Run some basic tests:

    $ lein test

## License

Copyright © 2016 Stephen M. Hopper

Distributed under the Eclipse Public License either version 1.0 or (at
your option) any later version.
