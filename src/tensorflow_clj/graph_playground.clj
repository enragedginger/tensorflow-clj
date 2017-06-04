(ns tensorflow-clj.graph-playground
  (require [tensorflow-clj.proto-much :as proto-much]
           [tensorflow-clj.experimental :as exp]
           [flatland.protobuf.core :as proto]
           [tensorflow-clj.util :as util]
           [clojure.string :as str]))

;;Control dependencies start with a caret, apparently
(defn is-control-dep-name [name]
  (str/starts-with? name "^"))

(defn drop-caret [name]
  (subs name 1))

(defn add-caret [name]
  (str "^" name))

(defn assoc-not-empty [m k v]
  (if (and v (-> v empty? not))
    (assoc m k v)
    m))

(defn assoc-in-not-empty [m ks v]
  (if (and v (-> v empty? not))
    (assoc-in m ks v)
    m))

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

(defn build-node-name [op & {:keys [name prefix]}]
  (if name
    (let [base-name (or name op)
          fullname (if prefix
                     (str/join "/" [prefix base-name])
                     base-name)]
      fullname)
    (proto-much/gen-name op false)))

(defn build-node [op & {:keys [name inputs control-deps attr]}]
  (let [node {:op op
             :name (or name (build-node-name op))
             :inputs inputs
             :control-deps control-deps
             :attr (apply merge attr)}]
    (into {} (filter second node))))

(defn build-attr [k v]
  ;{:key k :value v}
  {k v})
(defn build-dims [dims]
  (mapv #(assoc {} :size %) dims))
(defn build-attr-value [value value-dtype dims]
  (let [attr (build-attr :value {
                                  :tensor {
                                           :dtype value-dtype
                                           :tensor_shape {}
                                           }
                                  })
        val-key (-> value-dtype proto-much/lookup-by-dtype :val-key)]
    (-> attr
      (assoc-in-not-empty [:value :tensor :tensor_shape :dim] (build-dims dims))
        (assoc-in [:value :tensor val-key] [value]))))
(defn build-attr-dtype [dtype]
  (build-attr :dtype { :type dtype }))
(defn build-attr-n [val]
  (build-attr :N { :i val }))
(defn build-attr-t [dtype]
  (build-attr :T { :type dtype }))
(defn build-attr-tidx [dtype]
  (build-attr :Tidx { :type dtype }))
(defn build-attr-tshape [dtype]
  (build-attr :Tshape { :type dtype }))
(defn build-attr-out-type [dtype]
  (build-attr :out_type { :type dtype }))
(defn build-attr-axis [val]
  (build-attr :axis { :i val }))
(defn build-attr-index [dtype]
  (build-attr :Index { :type dtype }))
(defn build-attr-shape [dims]
  (build-attr :shape {:shape {
                              :dim (build-dims dims)
                              }}))

(defn build-node-placeholder [dtype & {:keys [name prefix]}]
  (let [op "Placeholder"
        attr-dtype (build-attr-dtype dtype)
        ;;todo attr-shape?!?!?!
        fullname (build-node-name op :name name :prefix prefix)]
    (build-node op :name fullname :attr [attr-dtype])))

(defn build-node-const [value value-dtype dims & {:keys [name prefix]}]
  (let [op "Const"
        attr-dtype (build-attr-dtype value-dtype)
        attr-value (build-attr-value value value-dtype dims)
        fullname (build-node-name op :name name :prefix prefix)
        base (build-node op :name fullname :attr [attr-dtype attr-value])]
    base))

(defn build-node-variable [dims value-dtype & {:keys [name prefix]}]
  (let [op "VariableV2"
        fullname (build-node-name op :name name :prefix prefix)
        attr-dtype (build-attr-dtype value-dtype)
        attr-shape (build-attr-shape dims)
        base (build-node op :name fullname :attr [attr-dtype attr-shape])]
    base))

(defn find-dtype [attr]
  (or (-> attr :T :type)
      (-> attr :dtype :type)
      (-> attr :value :tensor :dtype)))

(defn build-node-assign [variable value]
  (let [op "Assign"
        fullname (str/join "/" [(-> variable :name) "Assign"])
        inputs (mapv :name [variable value])
        attr-t (build-attr-t (-> variable :attr find-dtype))
        base (build-node op :name fullname :inputs inputs :attr [attr-t])]
    base))

(defn build-node-identity [target]
  (let [op "Identity"
        fullname (str/join "/" [(-> target :name) "read"])
        attr-t (-> target :attr find-dtype build-attr-t)
        inputs [(-> target :name)]]
    (build-node op :name fullname :inputs inputs :attr [attr-t])))

(defn def-tensor-nodes [name value dtype dims]
  (let [target (build-node-variable dims dtype :name name)
        value-node (build-node-const value dtype dims :name "zeros" :prefix name)
        assign (build-node-assign target value-node)
        identity (build-node-identity target)]
    [target value-node assign identity]))

(defn build-node-matmul [x y]
  (let [op "MatMul"
        inputs (mapv :name [x y])
        attr-t (-> x :attr find-dtype build-attr-t)]
    (build-node op :inputs inputs :attr [attr-t])))

(defn build-node-add [x y]
  (let [op "Add"
        inputs (mapv :name [x y])
        attr-t (-> x :attr find-dtype build-attr-t)]
    (build-node op :inputs inputs :attr [attr-t])))

(defn build-node-sub [x y]
  (let [op "Sub"
        inputs (mapv :name [x y])
        attr-t (-> x :attr find-dtype build-attr-t)]
    (build-node op :inputs inputs :attr [attr-t])))

(defn build-node-slice [input-node begin-node size-node]
  (let [op "Slice"
        inputs (mapv :name [input-node begin-node size-node])
        attr-index (build-attr-index "DT_INT32")
        attr-t (build-attr-t "DT_INT32")]
    (build-node op :inputs inputs :attr [attr-index attr-t])))

(defn build-node-concat-v2 [value-nodes axis-node]
  (let [op "ConcatV2"
        inputs (mapv :name (concat value-nodes [axis-node]))
        attr-n (build-attr-n (count value-nodes))
        attr-t (build-attr-t "DT_INT32")
        attr-tidx (build-attr-tidx "DT_INT32")]
    (build-node op :inputs inputs :attr [attr-n attr-t attr-tidx])))

(defn build-node-reshape [tensor-node shape-node]
  (let [op "Reshape"
        inputs (mapv :name [tensor-node shape-node])
        attr-t (-> tensor-node :attr find-dtype build-attr-t)
        attr-tshape (build-attr-tshape "DT_INT32")]
    (build-node op :inputs inputs :attr [attr-t attr-tshape])))

(defn build-node-shape [input-node]
  (let [op "Shape"
        inputs (mapv :name [input-node])
        attr-t (-> input-node :attr find-dtype build-attr-t)
        attr-out-type (build-attr-out-type "DT_INT32")]
    (build-node op :inputs inputs :attr [attr-t attr-out-type])))

(defn build-node-pack [input-nodes]
  (let [op "Pack"
        inputs (mapv :name input-nodes)
        attr-n (build-attr-n (count input-nodes))
        attr-t (build-attr-t "DT_INT32")
        attr-axis (build-attr-axis 0)]
    (build-node op :inputs inputs :attr [attr-n attr-t attr-axis])))

(defn build-node-softmax-cross-entropy-with-logits [labels-node logits-node]
  (let [op "SoftmaxCrossEntropyWithLogits"
        inputs (mapv :name [labels-node logits-node])
        attr-t (-> labels-node :attr find-dtype build-attr-t)]
    (build-node op :inputs inputs :attr [attr-t])))

(defn build-node-reduce-mean [input-node reduction-indices-node]
  (let [op "Mean"
        inputs (mapv :name [input-node reduction-indices-node])
        attr-t (-> input-node :attr find-dtype build-attr-t)
        ;;todo build keep_dims attr?
        attr-tidx (build-attr-tidx "DT_INT32")]
    (build-node op :inputs inputs :attr [attr-t attr-tidx])))

(defn build-node-apply-gradient-descent [input-node alpha-node delta-node]
  (let [op "ApplyGradientDescent"
        inputs (mapv :name [input-node alpha-node delta-node])
        attr-t (-> input-node :attr find-dtype build-attr-t)]
    (build-node op :inputs inputs :attr [attr-t])))

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

(let [y-mx-b-nodes (build-nodes-y-mx-b)
      placeholder-node (build-node-placeholder "DT_FLOAT" :name "y_hat")
      softmax-nodes (build-nodes-softmax-cross-entropy-with-logits (last y-mx-b-nodes) placeholder-node)
      reduce-mean-nodes (build-nodes-reduce-mean (last softmax-nodes))
      nodes (concat y-mx-b-nodes softmax-nodes reduce-mean-nodes)
      tf-nodes (map clj-node->tensorflow-node nodes)
      graph {:node tf-nodes
             :versions {:producer 21}}]
  (proto/protobuf-load proto-much/proto-graph-def
                       (proto/protobuf-dump proto-much/proto-graph-def graph))
  graph)

;;todo build these for reals
(defn build-optimizer [])
(defn build-gradient-descent-optimizer [])
(defn build-loop [loop-node times])
(defn build-train-next-batch [])
(defn build-prediction-check [])
(defn build-equal-check [])
(defn build-cast [node dtype])
(defn build-ApplyGradientDescent [variable learning-rate gradient-control])

(def linreg-graph (proto/protobuf-load proto-much/proto-graph-def (util/slurp-binary "misc/linreg.pb")))
(-> linreg-graph :node count)
(mapv :name (-> linreg-graph :node))
(filter #(= "Identity" (:op %)) (-> linreg-graph :node))
(map #(str (:name %) " " (:op %) " " (:input %)) (-> linreg-graph :node))

(def addconst-graph (proto/protobuf-load proto-much/proto-graph-def (util/slurp-binary "misc/addconst.pb")))
(-> addconst-graph :node count)

(def nonsense-graph (proto/protobuf-load proto-much/proto-graph-def (util/slurp-binary "misc/nonsense.pb")))

(def mnist-simple-graph (proto/protobuf-load proto-much/proto-graph-def (util/slurp-binary "misc/mnist_simple.pbtxt")))
(def mnist-meta-graph (proto/protobuf-load proto-much/proto-meta-graph-def (util/slurp-binary "misc/mnist_simple.model.meta")))

(-> mnist-meta-graph keys)

(defn parse-node-ref [node-ref]
  (let [[name output] (str/split node-ref #":")]
    (-> {}
        (assoc-not-empty :name name)
        (assoc-not-empty :output output))))

(defn parse-trainable-var [entry]
  (let [split-entry (-> entry
                        proto-much/byte-string-to-string
                        (str/replace "\n" "")
                        (str/replace "\b" "")
                        (str/split #"[\u0003\u0012\u001A]"))
        [var assign identity] (map parse-node-ref (remove empty? split-entry))]
    {:variable var
     :assign assign
     :identity identity}))

(defn parse-trainable-vars [graph]
  (->> graph
       :collection-def
       (filter #(= "trainable_variables" (:key %)))
       first
       :value
       :bytes-list
       :value
       (map parse-trainable-var)))

(parse-trainable-vars mnist-meta-graph)

(def thing (-> nonsense-graph :node second :attr second :value :tensor :string-val first proto-much/byte-string-to-string))

(time
  (exp/exec-graph-sess-fn
    (fn [graph session]
      (let [nodes (build-nodes-y-mx-b)
            out-node-name (-> nodes last :name)
            tf-nodes (map clj-node->tensorflow-node nodes)
            graph-def {:node tf-nodes
                       :versions {:producer 21}}
            graph-bytes (proto/protobuf-dump proto-much/proto-graph-def graph-def)]
        (.importGraphDef graph graph-bytes)
        (exp/run-graph-thing session {:x [[5.0 12.0]
                                          [2.5 3.4]]
                                      :W [[6.0 1.3]
                                          [0.0 0.0]]
                                      :b [[0.0]]}
                             out-node-name)
        ))))

(time
  (exp/exec-graph-sess-fn
    (fn [graph session]
      (let [y-mx-b-nodes (build-nodes-y-mx-b)
            out-node-1 (-> y-mx-b-nodes last)
            placeholder-node (build-node-placeholder "DT_FLOAT" :name "y_hat")
            softmax-nodes (build-nodes-softmax-cross-entropy-with-logits out-node-1 placeholder-node)
            out-node-2 (-> softmax-nodes last)
            nodes (concat y-mx-b-nodes softmax-nodes)
            tf-nodes (map clj-node->tensorflow-node nodes)
            graph-def {:node tf-nodes
                       :versions {:producer 21}}
            graph-bytes (proto/protobuf-dump proto-much/proto-graph-def graph-def)]
        (.importGraphDef graph graph-bytes)
        (exp/run-graph-thing session {:x [[5.0 12.0]
                                          [2.5 3.4]]
                                      :W [[6.0 1.3]
                                          [0.0 0.0]]
                                      :b [[2.0]]
                                      :y_hat [[5.3 8.5]
                                              [900.24 9.94]]}
                             (-> out-node-1 :name)
                             (-> out-node-2 :name))
        ))))

(time
  (exp/exec-graph-sess-fn
    (fn [graph session]
      (let [y-mx-b-nodes (build-nodes-y-mx-b)
            out-node-1 (-> y-mx-b-nodes last)
            placeholder-node (build-node-placeholder "DT_FLOAT" :name "y_hat")
            softmax-nodes (build-nodes-softmax-cross-entropy-with-logits out-node-1 placeholder-node)
            out-node-2 (-> softmax-nodes last)
            reduce-mean-nodes (build-nodes-reduce-mean out-node-2)
            out-node-3 (-> reduce-mean-nodes last)
            nodes (concat y-mx-b-nodes softmax-nodes reduce-mean-nodes)
            tf-nodes (map clj-node->tensorflow-node nodes)
            graph-def {:node tf-nodes
                       :versions {:producer 21}}
            graph-bytes (proto/protobuf-dump proto-much/proto-graph-def graph-def)]
        (.importGraphDef graph graph-bytes)
        (exp/run-graph-thing session {:x [[5.0 12.0]
                                          [2.5 3.4]]
                                      :W [[6.0 1.3]
                                          [0.0 0.0]]
                                      :b [[2.0]]
                                      :y_hat [[5.3 3.5]
                                              [0.24 3.94]]}
                             (-> out-node-1 :name)
                             (-> out-node-2 :name)
                             (-> out-node-3 :name))
        ))))
