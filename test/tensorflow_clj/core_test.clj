(ns tensorflow-clj.core-test
  (:require [clojure.test :refer :all]
            [tensorflow-clj.core :refer :all]))

(deftest scalar-tensor
  (testing "Scalar tensor"
    (let [t (org.tensorflow.Tensor/create 123.0)]
      (is (= org.tensorflow.DataType/DOUBLE (.dataType t)))
      (is (= 0 (.numDimensions t)))
      (is (= [] (vec (.shape t)))))))

(deftest vector-tensor
  (testing "Vector tensor"
    (let [t (org.tensorflow.Tensor/create (into-array [1.0 2.0 3.0]))]
      (is (= org.tensorflow.DataType/DOUBLE (.dataType t)))
      (is (= 1 (.numDimensions t)))
      (is (= [3] (vec (.shape t)))))))

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
      (let [t (run-graph {} :Const)]
        (is (= org.tensorflow.Tensor (type t)))
        (is (= [] (vec (.shape t))))
        (is (= 123.0 (.floatValue t)))))))
