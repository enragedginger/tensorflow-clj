(ns tensorflow-clj.core-test
  (:require [clojure.test :refer :all]
            [tensorflow-clj.core :refer :all]))

(deftest scalar-tensor
  (testing "Scalar tensor creation"
    (let [t (org.tensorflow.Tensor/create 123.0)]
      (is (= org.tensorflow.DataType/DOUBLE (.dataType t))))))
