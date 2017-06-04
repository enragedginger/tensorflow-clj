(ns tensorflow-clj.proto-much
  (:require [flatland.protobuf.core :as proto]
            [tensorflow-clj.util :as util]
            [random-string.core :as randy-str])
  (:import
    (org.tensorflow.framework.GraphDef)))

(def proto-meta-graph-def (proto/protodef org.tensorflow.framework.MetaGraphDef))
(def proto-graph-def (proto/protodef org.tensorflow.framework.GraphDef))
(def graph-node (proto/protodef org.tensorflow.framework.NodeDef))

(defn gen-name [prefix preserve?]
  (if preserve?
    prefix
    (str prefix "_" (randy-str/string 16))))

(defn convert-op-key [op-key]
  (-> op-key
      name
      clojure.string/lower-case
      clojure.string/capitalize))

(defn build-node-def [node-name-map inputs-map key entry]
  (let [attr (case (-> entry :op)
               :placeholder [{:key "dtype", :value {:type "DT_FLOAT"}} {:key "shape", :value {:shape {}}}]
               :mul [{:key "T", :value {:type "DT_FLOAT"}}]
               :add [{:key "T", :value {:type "DT_FLOAT"}}]
               )]
    {
     :name (get node-name-map key)
     :op (-> entry :op convert-op-key)
     :input (map #(get node-name-map %) (get inputs-map key))
     :attr attr
     }
    ))

(defn build-inputs-map [inputs]
  (let [keys (map second inputs)]
    (zipmap keys (map first inputs))))

(defn build-node-name-map [node-keys inputs outputs]
  (let [preservation-set (into #{} (concat inputs outputs))
        node-names (map #(gen-name (name %) (contains? preservation-set %)) node-keys)]
    (zipmap node-keys node-names)))

(defn build-tf-graph [{:keys [inputs outputs mappings node-defs] :as graph-def}]
  (let [
        node-keys (keys node-defs)
        node-name-map (build-node-name-map node-keys inputs outputs)
        inputs-map (build-inputs-map mappings)
        node-defs (map #(apply (partial build-node-def node-name-map inputs-map) %) node-defs)]
    {:node node-defs
     :versions {:producer 21}}))

(defn graph-to-bytes [graph]
  (let [proto-graph (apply
                      (partial proto/protobuf proto-graph-def)
                      (->> graph
                           (into [])
                           (apply concat)))]
    (proto/protobuf-dump proto-graph)))

(defn byte-string-to-string [^com.google.protobuf.ByteString$LiteralByteString byte-string-literal]
  (-> byte-string-literal .toStringUtf8))

;(exp/exec-graph-sess-fn
;  (fn [graph session]
;    ;(.importGraphDef graph (util/slurp-binary "misc/addconst.pb"))
;    ;(.importGraphDef graph (proto/protobuf-dump addconst-graph))
;    (.importGraphDef graph (proto-much/graph-to-bytes (proto-much/build-tf-graph example-graph)))
;    (exp/run-graph-thing session  {:in1 (float 3.0) :in2 (float 3.0)} :mul)))
