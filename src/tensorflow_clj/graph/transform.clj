(ns tensorflow-clj.graph.transform
  (require [clojure.string :as str]
           [tensorflow-clj.util :refer [assoc-not-empty assoc-in-not-empty]]))

;;Control dependencies start with a caret, apparently
(defn is-control-dep-name [name]
  (str/starts-with? name "^"))

(defn drop-caret [name]
  (subs name 1))

(defn add-caret [name]
  (str "^" name))

;;Transforme node defs from Tensorflow into something suitable for us to play with (and back again) and don't
;;pretend to know everything about the structure / content of the map
(defn tensorflow-node->clj-node [node]
  (let [inputs (filter (complement is-control-dep-name) (:input node))
        control-deps (mapv drop-caret (filter is-control-dep-name (:input node)))
        converted-attrs (into {} (mapv #(vec [(:key %) (:value %)]) (:attr node)))]
    (-> node
        (dissoc :input)
        (assoc :attr converted-attrs)
        (assoc-not-empty :inputs inputs)
        (assoc-not-empty :control-deps control-deps))))

(defn clj-node->tensorflow-node [node]
  (let [input (concat (:inputs node) (mapv add-caret (:control-deps node)))
        converted-attrs (into [] (mapv (fn [[k v]] (apply hash-map [:key (name k) :value v])) (:attr node)))]
    (-> node
        (dissoc :inputs :control-deps)
        (assoc :attr converted-attrs)
        (assoc-not-empty :input input))))