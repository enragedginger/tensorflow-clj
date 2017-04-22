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
      (let [[t] (run-graph {} :Const)]
        (is (= org.tensorflow.Tensor (type t)))
        (is (= [] (vec (.shape t))))
        (is (= 123.0 (.floatValue t)))))))

(deftest protobuf-feed
  (testing "Variable feed to loaded graph"
    (with-graph-file "misc/addconst.pb"
      (let [[t] (run-graph {:Placeholder (tensor (float 123.0))} :mul)]
        (is (= org.tensorflow.Tensor (type t)))
        (is (= [] (vec (.shape t))))
        (is (= 369.0 (.floatValue t)))))))

(deftest matrix-feed
  (testing "Matrix fed to loaded graph"
    (with-graph-file "misc/addconst.pb"
      (let [[t] (run-graph {:Placeholder (tensor [[1 2] [3 4]])} :mul)]
        (is (= org.tensorflow.Tensor (type t)))
        (is (= [2 2] (vec (.shape t))))))))
