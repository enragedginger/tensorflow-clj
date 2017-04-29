(ns tensorflow-clj.experimental-test
  (:require [clojure.test :refer :all]
            [tensorflow-clj.core :refer :all]
            [tensorflow-clj.experimental :refer :all]
            [tensorflow-clj.util :refer :all]))

(def x-train [1.  2.  3.  4.])
(def y-train [0. -1. -2. -3.])

(deftest linreg-one-pass
  (testing "Linear regression (one pass)"
    (exec-graph-sess-fn
      (fn [graph session]
        (load-graph! graph "misc/linreg.pb")
        (run-graph-thing session {:init nil})
        (let [[[a b c d]] (run-graph-thing session {:x x-train} :linear_model)]
          (is (approx= 0.0 a))
          (is (approx= 0.3 b))
          (is (approx= 0.6 c))
          (is (approx= 0.9 d)))))))

(deftest linreg-one-loss
  (testing "Linear regression (one loss)"
    (exec-graph-sess-fn
      (fn [graph session]
        (load-graph! graph "misc/linreg.pb")
        (run-graph-thing session {:init nil})
        (let [[loss] (run-graph-thing session {:x x-train :y y-train} :loss)]
          (is (approx= 23.66 loss)))))))

(deftest linreg-fixed-vars
  (testing "Linear regression (fixed variables)"
    (exec-graph-sess-fn
      (fn [graph session]
        (load-graph! graph "misc/linreg.pb")
        (run-graph-thing session {:fixW nil :fixb nil})
        (let [[loss] (run-graph-thing session {:x x-train :y y-train} :loss)]
          (is (approx= 0.0 loss)))))))

(deftest linreg-graph-iterations
  (testing "Linear regression (1000 iterations)"
    (exec-graph-sess-fn
      (fn [graph session]
        (load-graph! graph "misc/linreg.pb")
        (run-graph-thing session {:init nil})
        (dotimes [i 1000]
          (run-graph-thing session {:x x-train :y y-train :train nil}))
        (let [[[W] [b]] (run-graph-thing session {} :W :b)]
          (is (approx= -0.9999 W))
          (is (approx= 0.99999 b)))))))
