(ns tensorflow-clj.graph.proto-much
  (:require [flatland.protobuf.core :as proto]
            [tensorflow-clj.util :as util]
            [random-string.core :as randy-str])
  (:import
    (org.tensorflow.framework.GraphDef)))

(def proto-meta-graph-def (proto/protodef org.tensorflow.framework.MetaGraphDef))
(def proto-graph-def (proto/protodef org.tensorflow.framework.GraphDef))
(def graph-node (proto/protodef org.tensorflow.framework.NodeDef))

(defn graph-to-bytes [graph]
  (let [proto-graph (apply
                      (partial proto/protobuf proto-graph-def)
                      (->> graph
                           (into [])
                           (apply concat)))]
    (proto/protobuf-dump proto-graph)))

(defn byte-string-to-string [^com.google.protobuf.ByteString$LiteralByteString byte-string-literal]
  (-> byte-string-literal .toStringUtf8))
