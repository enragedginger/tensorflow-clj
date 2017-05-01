(ns tensorflow-clj.core-test
  (:require [clojure.test :refer :all]
            [tensorflow-clj.core :refer :all]
            [tensorflow-clj.experimental :refer :all]
            [tensorflow-clj.util :refer :all]))

(defmacro test-both-apis [graph-file & body]
  `(do
     (with-graph-file ~graph-file
       (letfn [(~'run-graph [& args#] (apply run-graph args#))]
         ~@body))
     (exec-graph-sess-fn
       (fn [graph# session#]
         (load-graph! graph# ~graph-file)
         (letfn [(~'run-graph [& args#] (apply run-graph-thing session# args#))]
           ~@body)))))

(deftest scalar-tensor
  (testing "Scalar tensor"
    (let [t (tensor 123.0)]
      (is (= org.tensorflow.DataType/FLOAT (.dataType t)))
      (is (= 0 (.numDimensions t)))
      (is (= [] (vec (.shape t)))))))

(deftest vector-tensor
  (testing "Vector tensor"
    (let [t (tensor [1.0 2.0 3.0])]
      (is (= org.tensorflow.DataType/FLOAT (.dataType t)))
      (is (= 1 (.numDimensions t)))
      (is (= [3] (vec (.shape t)))))))

(deftest matrix-tensor
  (testing "Matrix tensor"
    (let [t (tensor [[1.0 2.0 3.0]
                     [4.0 5.0 6.0]])]
      (is (= org.tensorflow.DataType/FLOAT (.dataType t)))
      (is (= 2 (.numDimensions t)))
      (is (= [2 3] (vec (.shape t)))))))

(deftest tensor-conversion
  (testing "Converting between tensors and core.matrix"
    (letfn [(test [x] (is (= x (tensor->clj (tensor x)))))]
      (test 123.0)
      (test [1.0 2.0 3.0])
      (test [[1.0 -2.0 3.0]
             [4.0 5.0 -6.0]])
      (test [[[1., 2., 3.]], [[7., 8., 9.]]])
      (test [[[[555.5]]]]))))

(deftest protobuf-session
  (testing "Session from Protocol Buffers file"
    (test-both-apis "misc/constant.pb"
      (let [[v] (run-graph {} :Const)]
        (is (= 123.0 v))))))

(deftest protobuf-feed
  (testing "Variable feed to loaded graph"
    (test-both-apis "misc/addconst.pb"
      (let [[v] (run-graph {:Placeholder (float 123.0)} :mul)]
        (is (= 369.0 v))))))

(deftest matrix-feed
  (testing "Matrix fed to loaded graph"
    (test-both-apis "misc/addconst.pb"
      (let [[v] (run-graph {:Placeholder [[1 2] [3 4]]} :mul)]
        (is (= v [[3.0 6.0] [9.0 12.0]]))))))

(deftest mulbymat-graph
  (testing "Multiplying variable by constant matrix"
    (test-both-apis "misc/mulbymat.pb"
      (let [[v] (run-graph {:Placeholder 5} :mul)]
        (is (= v [[5. 10.] [15. 20.]])))
      (let [[v] (run-graph {:Placeholder [[1. -1.] [2. -2.]]} :mul)]
        (is (= v [[1. -2.] [6. -8.]]))))))

(deftest mul2vars-graph
  (testing "Multiplying two variables"
    (test-both-apis "misc/mul2vars.pb"
      (let [[v] (run-graph {:a 4. :b 10.5} :mul)]
        (is (= v 42.0)))
      (let [[v] (run-graph {:a [[1. -1.] [2. -2.]]
                            :b [[1. 2.] [3. 4.]]}
                  :mul)]
        (is (= v [[1. -2.] [6. -8.]]))))))

(def x-train [1.  2.  3.  4.])
(def y-train [0. -1. -2. -3.])

(deftest linreg-one-pass
  (testing "Linear regression (one pass)"
    (test-both-apis "misc/linreg.pb"
      (run-graph {:init nil})
      (let [[[a b c d]] (run-graph {:x x-train} :linear_model)]
        (is (approx= 0.0 a))
        (is (approx= 0.3 b))
        (is (approx= 0.6 c))
        (is (approx= 0.9 d))))))

(deftest linreg-one-loss
  (testing "Linear regression (one loss)"
    (test-both-apis "misc/linreg.pb"
      (run-graph {:init nil})
      (let [[loss] (run-graph {:x x-train :y y-train} :loss)]
        (is (approx= 23.66 loss))))))

(deftest linreg-fixed-vars
  (testing "Linear regression (fixed variables)"
    (test-both-apis "misc/linreg.pb"
      (run-graph {:fixW nil :fixb nil})
      (let [[loss] (run-graph {:x x-train :y y-train} :loss)]
        (is (approx= 0.0 loss))))))

(deftest linreg-graph-iterations
  (testing "Linear regression (1000 iterations)"
    (test-both-apis "misc/linreg.pb"
      (run-graph {:init nil})
      (dotimes [i 1000]
        (run-graph {:x x-train :y y-train :train nil}))
      (let [[[W] [b] loss] (run-graph {:x x-train :y y-train} :W :b :loss)]
        (is (approx= -0.9999 W))
        (is (approx= 0.99999 b))
        (is (approx= 5.6999738e-11 loss))))))
