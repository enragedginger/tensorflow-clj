(ns tensorflow-clj.experimental-test
  (:require [clojure.test :refer :all]
            [tensorflow-clj.core :refer :all]
            [tensorflow-clj.experimental :refer :all]
            [tensorflow-clj.util :refer :all]))

(deftest linreg-graph
  (testing "Linear regression"
    (let [x-train [1. 2. 3. 4.]
          y-train [0. -1. -2. -3.]]
      (exec-graph-sess-fn
        (fn [graph session]
          (load-graph! graph "misc/linreg.pb")
          (let [[v] (run-graph-thing session {:x x-train :y y-train :W [-1.] :b [1.]}
                               :loss)]
            (is (= v 0.0))))))
    (let [W -1.
          b 1.
          x-train (into [] (map (fn [i] (* 1000 (Math/random))) (range 1000)))
          y-train (map #(+ (* W %) b) x-train)]
      (exec-graph-sess-fn
        (fn [graph session]
          (load-graph! graph "misc/linreg.pb")
          (let [[v] (run-graph-thing session {:x x-train :y y-train :W [W] :b [b]}
                               :loss)]
            (is (approx= 0.0 v))))))))
