(ns tensorflow-clj.graph-playground
  (require [tensorflow-clj.graph.proto-much :as proto-much]
           [tensorflow-clj.experimental :as exp]
           [flatland.protobuf.core :as proto]
           [tensorflow-clj.util :as util]
           [tensorflow-clj.graph.node_defs :refer :all]
           [tensorflow-clj.graph.transform :refer :all]
           [tensorflow-clj.util :refer [assoc-not-empty assoc-in-not-empty]]
           [clojure.string :as str]))
;TODO this is a playground namespace for now
;pieces will be hacked on here in isolation and then moved out to other namespaces
;where they can be re-used once they are considered semi-stable
;much of this code is commented out as I'm just running it once here or there
;during development. anything that I think *might* be useful will be thrown into a function
;any functions that prove to have some utility will be grouped and moved to their own ns

(defn def-tensor-nodes [name value dtype dims]
  (let [target (build-node-variable dims dtype :name name)
        value-node (build-node-const value dtype dims :name "zeros" :prefix name)
        assign (build-node-assign target value-node)
        identity (build-node-identity target)]
    [target value-node assign identity]))

(defn build-nodes-y-mx-b []
  (let [x-nodes (def-tensor-nodes "x" 0.0 "DT_FLOAT" [10 784])
        W-nodes (def-tensor-nodes "W" 0.0 "DT_FLOAT" [784 10])
        b-nodes (def-tensor-nodes "b" 0.0 "DT_FLOAT" [10])
        matmul-node (build-node-matmul (last x-nodes) (last W-nodes))
        add-node (build-node-add matmul-node (last b-nodes))]
    (concat x-nodes W-nodes b-nodes [matmul-node add-node])))

(defn build-nodes-softmax-cross-entropy-with-logits [input-node placeholder-node]
  (let [rank-0-node (build-node-const 2 "DT_INT32" [])
        shape-0-node (build-node-shape input-node)
        rank-1-node (build-node-const 2 "DT_INT32" [])
        shape-1-node (build-node-shape input-node)
        sub-y-node (build-node-const 1 "DT_INT32" [])
        sub-node (build-node-sub rank-0-node sub-y-node)
        slice-begin-node (build-node-pack [sub-node])
        slice-size-node (build-node-const 1 "DT_INT32" [1])
        slice-node (build-node-slice shape-1-node slice-begin-node slice-size-node)
        concat-values-0-node (build-node-const -1 "DT_INT32" [1])
        concat-axis-node (build-node-const 0 "DT_INT32" [])
        concat-node (build-node-concat-v2 [concat-values-0-node slice-node] concat-axis-node)
        reshape-node (build-node-reshape input-node concat-node)
        rank-2-node (build-node-const 2 "DT_INT32" [])
        shape-2-node (build-node-shape placeholder-node)
        sub-1-y-node (build-node-const 1 "DT_INT32" [])
        sub-1-node (build-node-sub rank-2-node sub-1-y-node)
        slice-1-begin-node (build-node-pack [sub-1-node])
        slice-1-size-node (build-node-const 1 "DT_INT32" [1])
        slice-1-node (build-node-slice shape-2-node slice-1-begin-node slice-1-size-node)
        concat-1-values-0-node (build-node-const -1 "DT_INT32" [1])
        concat-1-axis-node (build-node-const 0 "DT_INT32" [])
        concat-1-node (build-node-concat-v2 [concat-1-values-0-node slice-1-node] concat-1-axis-node)
        reshape-1-node (build-node-reshape placeholder-node concat-1-node)
        cross-entropy-node (build-node-softmax-cross-entropy-with-logits reshape-node reshape-1-node)]
    [placeholder-node rank-0-node shape-0-node rank-1-node shape-1-node sub-y-node
     sub-node slice-begin-node slice-size-node slice-node concat-values-0-node
     concat-axis-node concat-node reshape-node rank-2-node shape-2-node
     sub-1-y-node sub-1-node slice-1-begin-node slice-1-size-node slice-1-node
     concat-1-values-0-node concat-1-axis-node concat-1-node reshape-1-node cross-entropy-node
     ]))

(defn build-nodes-reduce-mean [input-node]
  (let [reduction-indices-node (build-node-const 0 "DT_INT32" [1])
        reduce-mean-node (build-node-reduce-mean input-node reduction-indices-node)]
    [reduction-indices-node reduce-mean-node]))

;;todo build these for reals
(defn build-loop [loop-node times])
(defn build-train-next-batch [])
(defn build-prediction-check [])
(defn build-equal-check [])
(defn build-cast [node dtype])
(defn build-ApplyGradientDescent [variable learning-rate gradient-control])

;(def linreg-graph (proto/protobuf-load proto-much/proto-graph-def (util/slurp-binary "misc/linreg.pb")))
;(-> linreg-graph :node count)
;(mapv :name (-> linreg-graph :node))
;(filter #(= "Identity" (:op %)) (-> linreg-graph :node))
;(map #(str (:name %) " " (:op %) " " (:input %)) (-> linreg-graph :node))
;(def addconst-graph (proto/protobuf-load proto-much/proto-graph-def (util/slurp-binary "misc/addconst.pb")))
;(-> addconst-graph :node count)
;(def nonsense-graph (proto/protobuf-load proto-much/proto-graph-def (util/slurp-binary "misc/nonsense.pb")))
;(def mnist-simple-graph (proto/protobuf-load proto-much/proto-graph-def (util/slurp-binary "misc/mnist_simple.pbtxt")))
;(def mnist-meta-graph (proto/protobuf-load proto-much/proto-meta-graph-def (util/slurp-binary "misc/mnist_simple.model.meta")))
;(-> mnist-meta-graph keys)

(defn parse-node-ref [node-ref]
  (let [[name output] (str/split node-ref #":")]
    (-> {}
        (assoc-not-empty :name name)
        (assoc-not-empty :output output))))

(defn parse-variable [trainable? entry]
  (let [split-entry (-> entry
                        proto-much/byte-string-to-string
                        (str/replace "\n" "")
                        (str/replace "\b" "")
                        (str/split #"[\u0003\u0012\u001A]"))
        [var assign identity] (map parse-node-ref (remove empty? split-entry))]
    {:variable var
     :assign assign
     :identity identity
     :trainable? trainable?}))

(defn parse-trainable-vars [graph]
  (->> graph
       :collection-def
       (filter #(= "trainable_variables" (:key %)))
       first
       :value
       :bytes-list
       :value
       (map (partial parse-variable true))))

;(parse-trainable-vars mnist-meta-graph)
;(def thing (-> nonsense-graph :node second :attr second :value :tensor :string-val first proto-much/byte-string-to-string))

;(let [y-mx-b-nodes (build-nodes-y-mx-b)
;      placeholder-node (build-node-placeholder "DT_FLOAT" :name "y_hat")
;      softmax-nodes (build-nodes-softmax-cross-entropy-with-logits (last y-mx-b-nodes) placeholder-node)
;      reduce-mean-nodes (build-nodes-reduce-mean (last softmax-nodes))
;      nodes (concat y-mx-b-nodes softmax-nodes reduce-mean-nodes)
;      tf-nodes (map clj-node->tensorflow-node nodes)
;      graph {:node tf-nodes
;             :versions {:producer 21}}]
;  (proto/protobuf-load proto-much/proto-graph-def
;                       (proto/protobuf-dump proto-much/proto-graph-def graph))
;  graph)

;(time
;  (exp/exec-graph-sess-fn
;    (fn [graph session]
;      (let [nodes (build-nodes-y-mx-b)
;            out-node-name (-> nodes last :name)
;            tf-nodes (map clj-node->tensorflow-node nodes)
;            graph-def {:node tf-nodes
;                       :versions {:producer 21}}
;            graph-bytes (proto/protobuf-dump proto-much/proto-graph-def graph-def)]
;        (.importGraphDef graph graph-bytes)
;        (exp/run-graph-thing session {:x [[5.0 12.0]
;                                          [2.5 3.4]]
;                                      :W [[6.0 1.3]
;                                          [0.0 0.0]]
;                                      :b [[0.0]]}
;                             out-node-name)
;        ))))
;
;(time
;  (exp/exec-graph-sess-fn
;    (fn [graph session]
;      (let [y-mx-b-nodes (build-nodes-y-mx-b)
;            out-node-1 (-> y-mx-b-nodes last)
;            placeholder-node (build-node-placeholder "DT_FLOAT" :name "y_hat")
;            softmax-nodes (build-nodes-softmax-cross-entropy-with-logits out-node-1 placeholder-node)
;            out-node-2 (-> softmax-nodes last)
;            nodes (concat y-mx-b-nodes softmax-nodes)
;            tf-nodes (map clj-node->tensorflow-node nodes)
;            graph-def {:node tf-nodes
;                       :versions {:producer 21}}
;            graph-bytes (proto/protobuf-dump proto-much/proto-graph-def graph-def)]
;        (.importGraphDef graph graph-bytes)
;        (exp/run-graph-thing session {:x [[5.0 12.0]
;                                          [2.5 3.4]]
;                                      :W [[6.0 1.3]
;                                          [0.0 0.0]]
;                                      :b [[2.0]]
;                                      :y_hat [[5.3 8.5]
;                                              [900.24 9.94]]}
;                             (-> out-node-1 :name)
;                             (-> out-node-2 :name))
;        ))))
;
;(time
;  (exp/exec-graph-sess-fn
;    (fn [graph session]
;      (let [y-mx-b-nodes (build-nodes-y-mx-b)
;            out-node-1 (-> y-mx-b-nodes last)
;            placeholder-node (build-node-placeholder "DT_FLOAT" :name "y_hat")
;            softmax-nodes (build-nodes-softmax-cross-entropy-with-logits out-node-1 placeholder-node)
;            out-node-2 (-> softmax-nodes last)
;            reduce-mean-nodes (build-nodes-reduce-mean out-node-2)
;            out-node-3 (-> reduce-mean-nodes last)
;            nodes (concat y-mx-b-nodes softmax-nodes reduce-mean-nodes)
;            tf-nodes (map clj-node->tensorflow-node nodes)
;            graph-def {:node tf-nodes
;                       :versions {:producer 21}}
;            graph-bytes (proto/protobuf-dump proto-much/proto-graph-def graph-def)]
;        (.importGraphDef graph graph-bytes)
;        (exp/run-graph-thing session {:x [[5.0 12.0]
;                                          [2.5 3.4]]
;                                      :W [[6.0 1.3]
;                                          [0.0 0.0]]
;                                      :b [[2.0]]
;                                      :y_hat [[5.3 3.5]
;                                              [0.24 3.94]]}
;                             (-> out-node-1 :name)
;                             (-> out-node-2 :name)
;                             (-> out-node-3 :name))
;        ))))
