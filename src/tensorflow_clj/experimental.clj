(ns tensorflow-clj.experimental
  (:require [clojure.core.matrix :as matrix]
            [tensorflow-clj.core :as core]
            [tensorflow-clj.util :as util]))

(defn exec-graph-fn [graph-fn]
  (let [graph (org.tensorflow.Graph.)]
    (try
      (graph-fn graph)
      (finally
        (.close graph)))))

(defn exec-graph-sess-fn [graph-sess-fn]
  (exec-graph-fn
    (fn [graph]
      (let [session (org.tensorflow.Session. graph)]
        (try
          (graph-sess-fn graph session)
          (finally
            (.close session)))))))

(defn load-graph! [graph filename]
  (.importGraphDef graph (util/slurp-binary filename)))

(defn run-graph-thing [session feed-ops & fetch-ops]
  (let [runner (.runner session)]
    (doseq [[feed-op feed-value] feed-ops]
      (if feed-value
        (.feed runner (name feed-op) (core/tensor feed-value))
        (.addTarget runner (name feed-op))))
    (doseq [fetch-op fetch-ops]
      (.fetch runner (name fetch-op)))
    (vec (map core/tensor->clj (.run runner)))))
