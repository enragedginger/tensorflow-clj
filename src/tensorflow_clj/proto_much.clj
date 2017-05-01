(ns tensorflow-clj.proto-much
  (:require [flatland.protobuf.core :as proto]
            [tensorflow-clj.experimental :as exp]
            [tensorflow-clj.util :as util]
            [random-string.core :as randy-str])
  (:import
    (org.tensorflow.framework.*)
    (org.tensorflow.framework.GraphDef)))

(def proto-graph-def (proto/protodef org.tensorflow.framework.GraphDef))
(def graph-node (proto/protodef org.tensorflow.framework.NodeDef))

(defn write-node [node]
  (apply
    (partial proto/protobuf graph-node)
    (->> node
         (into [])
         (apply concat))))

(def linreg-graph (proto/protobuf-load graph (util/slurp-binary "misc/linreg.pb")))
(-> linreg-graph :node count)
(map :name (-> linreg-graph :node))

(def addconst-graph (proto/protobuf-load graph (util/slurp-binary "misc/addconst.pb")))
(-> addconst-graph :node count)

(defn gen-name [prefix]
  ;(str prefix "_" (randy-str/string 16))
  prefix)

(defn convert-op-key [op-key]
  (-> op-key
      name
      clojure.string/lower-case
      clojure.string/capitalize))

(def example-graph
  {
   :inputs [:in1 :in2]
   :outputs [:mul]
   :mappings [
              [[:in1 :in2] :add1]
              [[:in1 :in2] :add2]
              [[:add1 :add2] :mul]
              ]
   :node-defs {
               :in1 {
                     :op :placeholder
                     :dtype :float
                     }
               :in2 {
                     :op :placeholder
                     :dtype :float
                     }
               :mul {
                     :op :mul
                     }
               :add1 {
                      :op :add
                      }
               :add2 {
                      :op :add
                      }
               }
   })


{:node [{:name "Const",
         :op "Const",
         :attr [{:key "value", :value {:tensor {:dtype :dt-float, :tensor-shape {}, :float-val [3.0]}}}
                {:key "dtype", :value {:type :dt-float}}]}
        {:name "Placeholder",
         :op "Placeholder",
         :attr [{:key "dtype", :value {:type :dt-float}} {:key "shape", :value {:shape {}}}]}
        {:name "mul", :op "Mul", :input ["Const" "Placeholder"], :attr [{:key "T", :value {:type :dt-float}}]}],
 :versions {:producer 21}}

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

(defn build-tf-graph [graph-def]
  (let [{:keys [inputs outputs mappings node-defs]} graph-def
        node-keys (keys node-defs)
        node-name-map (zipmap node-keys (map #(-> % name gen-name) node-keys))
        inputs-map (build-inputs-map mappings)
        node-defs (map #(apply (partial build-node-def node-name-map inputs-map) %) node-defs)]
    {:node node-defs
     :versions {:producer 21}}))

(build-tf-graph example-graph)

(defn write-graph [graph]
  (apply
    (partial proto/protobuf proto-graph-def)
    (->> graph
         (into [])
         (apply concat))))

(write-graph (build-tf-graph example-graph))
(write-graph {:node [{:name "Const",
                      :op "Const",
                      :attr [{:key "value", :value {:tensor {:dtype "DT_FLOAT", :tensor-shape {}, :float-val [3.0]}}}
                             {:key "dtype", :value {:type "DT_FLOAT"}}]}
                     {:name "Placeholder",
                      :op "Placeholder",
                      :attr [{:key "dtype", :value {:type "DT_FLOAT"}} {:key "shape", :value {:shape {}}}]}
                     {:name "mul", :op "Mul", :input ["Const" "Placeholder"], :attr [{:key "T", :value {:type "DT_FLOAT"}}]}],
              :versions {:producer 21}})

(exp/exec-graph-sess-fn
  (fn [graph session]
    ;(.importGraphDef graph (util/slurp-binary "misc/addconst.pb"))
    ;(.importGraphDef graph (proto/protobuf-dump addconst-graph))
    (.importGraphDef graph (proto/protobuf-dump (write-graph (build-tf-graph example-graph))))
    (exp/run-graph-thing session  {:in1 (float 3.0) :in2 (float 2.0)} :mul)))

(use 'clojure.repl)
(pst *e)
