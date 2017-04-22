(ns tensorflow-clj.core-test
  (:require [clojure.test :refer :all]
            [tensorflow-clj.core :refer :all]))

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

(deftest graph-variable
  (testing "Graph variable"
    (with-graph
      (variable :x))))

(deftest graph-constant
  (testing "Graph constant"
    (with-graph
      (constant :k 123.0))))

;; (deftest session
;;   (testing "Session"
;;     (with-graph
;;       (constant :k 123.0)
;;       (run-feed-and-fetch :k))))

(deftest protobuf-session
  (testing "Session from Protocol Buffers file"
    (with-graph-file "misc/constant.pb"
      (let [[v] (run-graph {} :Const)]
        (is (= 123.0 v))))))

(deftest protobuf-feed
  (testing "Variable feed to loaded graph"
    (with-graph-file "misc/addconst.pb"
      (let [[v] (run-graph {:Placeholder (float 123.0)} :mul)]
        (is (= 369.0 v))))))

(deftest matrix-feed
  (testing "Matrix fed to loaded graph"
    (with-graph-file "misc/addconst.pb"
      (let [[v] (run-graph {:Placeholder [[1 2] [3 4]]} :mul)]
        (is (= v [[3.0 6.0] [9.0 12.0]]))))))

(deftest mulbymat-graph
  (testing "Multiplying variable by constant matrix"
    (with-graph-file "misc/mulbymat.pb"
      (let [[v] (run-graph {:Placeholder 5} :mul)]
        (is (= v [[5. 10.] [15. 20.]])))
      (let [[v] (run-graph {:Placeholder [[1. -1.] [2. -2.]]} :mul)]
        (is (= v [[1. -2.] [6. -8.]]))))))

(deftest mul2vars-graph
  (testing "Multiplying two variables"
    (with-graph-file "misc/mul2vars.pb"
      (let [[v] (run-graph {:a 4. :b 10.5} :mul)]
        (is (= v 42.0))))))
