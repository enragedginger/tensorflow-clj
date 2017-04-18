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
    (let [g (org.tensorflow.Graph.)]
      (let [x (-> (.opBuilder g "Variable" "x")
                (.setAttr "dtype" org.tensorflow.DataType/DOUBLE)
                (.setAttr "shape" (org.tensorflow.Shape/scalar))
                (.build))]
        x))))
