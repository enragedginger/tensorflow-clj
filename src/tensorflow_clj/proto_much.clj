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
  ;(-> byte-string-literal .toByteArray String.)
  (-> byte-string-literal .toStringUtf8)
  )

(defn byte-string-to-string [^com.google.protobuf.ByteString$LiteralByteString byte-string-literal]
  ;(-> byte-string-literal .toByteArray String.)
  (-> byte-string-literal .toStringUtf8)
  )

(def tf-data-types
  [
   {
    :checker-fn float?
    :val-key :float-val
    :tf-enum-types #{"DT_FLOAT" "DT_DOUBLE" "DT_BFLOAT16"
                     "DT_FLOAT_REF" "DT_DOUBLE_REF" "DT_BFLOAT16_REF"}
    }
   {
    :checker-fn integer?
    :val-key :int-val
    :tf-enum-types #{"DT_INT32" "DT_UINT8" "DT_INT16" "DT_INT8" "DT_INT64" "DT_QINT8" "DT_QUINT8" "DT_QINT32" "DT_QINT16" "DT_QUINT16" "DT_UINT16"
                     "DT_INT32_REF" "DT_UINT8_REF" "DT_INT16_REF" "DT_INT8_REF" "DT_INT64_REF" "DT_QINT8_REF" "DT_QUINT8_REF"
                     "DT_QINT32_REF" "DT_QINT16_REF" "DT_QUINT16_REF" "DT_UINT16_REF"}
    }
   {
    :checker-fn (fn [x] (or (= x true) (= x false)))
    :val-key :bool-val
    :tf-enum-types #{"DT_BOOL" "DT_BOOL_REF"}
    }
   {
    :checker-fn string?
    :val-key :string-val
    :tf-enum-types #{"DT_STRING_REF"}
    }
   ]
  ;DT_COMPLEX64(8), DT_COMPLEX128(18), DT_HALF(19), DT_RESOURCE(20), DT_COMPLEX64_REF(108), DT_COMPLEX128_REF(118),
  ;DT_HALF_REF(119), DT_RESOURCE_REF(120), UNRECOGNIZED(-1)
  )

(defn lookup-by-dtype [dtype]
  (let [matches (filter #(contains? (:tf-enum-types %) dtype) tf-data-types)]
    (first matches)))

;(def example-graph
;  {
;   :inputs [:in1 :in2]
;   :outputs [:mul]
;   :mappings [
;              [[:in1 :in2] :add1]
;              [[:in1 :in2] :add2]
;              [[:add1 :add2] :mul]
;              ]
;   :node-defs {
;               :in1 { :op :placeholder :dtype :float }
;               :in2 { :op :placeholder :dtype :float }
;               :mul { :op :mul }
;               :add1 { :op :mul }
;               :add2 { :op :add }
;               }
;   })
;(proto-much/build-tf-graph example-graph)

;(exp/exec-graph-sess-fn
;  (fn [graph session]
;    ;(.importGraphDef graph (util/slurp-binary "misc/addconst.pb"))
;    ;(.importGraphDef graph (proto/protobuf-dump addconst-graph))
;    (.importGraphDef graph (proto-much/graph-to-bytes (proto-much/build-tf-graph example-graph)))
;    (exp/run-graph-thing session  {:in1 (float 3.0) :in2 (float 3.0)} :mul)))
