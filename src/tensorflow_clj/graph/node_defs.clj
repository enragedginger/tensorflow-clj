(ns tensorflow-clj.graph.node_defs
  (require [clojure.string :as str]
           [tensorflow-clj.graph.attributes :refer :all]
           [random-string.core :as randy-str]))

(defn gen-name [prefix preserve?]
  (if preserve?
    prefix
    (str prefix "_" (randy-str/string 16))))

(defn build-node-name [op & {:keys [name prefix]}]
  (if name
    (let [base-name (or name op)
          fullname (if prefix
                     (str/join "/" [prefix base-name])
                     base-name)]
      fullname)
    (gen-name op false)))

(defn build-node [op & {:keys [name inputs control-deps attr meta-attr]}]
  (let [node {:op op
              :name (or name (build-node-name op))
              :inputs inputs
              :control-deps control-deps
              :attr (apply merge attr)
              :meta-attr (apply merge meta-attr)}]
    (into {} (filter second node))))

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