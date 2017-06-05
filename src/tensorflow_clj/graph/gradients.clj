(ns tensorflow-clj.graph.gradients
  (require [tensorflow-clj.graph.variables :as variables]
           [tensorflow-clj.graph.node_defs :refer :all]
           [ubergraph.core :as uber]
           [ubergraph.alg :as uber-alg]))

(defn build-reverse-node-pointers [node]
  (let [node-name (:name node)]
    (map #(vec [node-name %]) (:inputs node))))

;;The list of ops that we can gradientize
(def gradientable-ops #{"Add", "MatMul", "Mean", "Reshape", "SoftmaxCrossEntropyWithLogits", "Sub"})

(defn find-gradientable-nodes [nodes]
  (let [pointers (mapcat identity (map build-reverse-node-pointers nodes))
        var-node-names (into #{} (map :variable (variables/find-variable-nodes nodes)))
        node-names (map :name (filter #(contains? gradientable-ops (:op %)) nodes))
        uber-graph (apply uber/digraph pointers)
        ;;find path from each gradientable node to a var and keep if it exists
        gradient-nodes (filter #(uber-alg/shortest-path
                                  uber-graph
                                  {:start-node % :end-nodes var-node-names}) node-names)]
    (uber/pprint uber-graph)
    gradient-nodes))

;;todo either use defined gradient or use symbolic gradient
;def _SymGrad(op, out_grads):
;"""Backprop through a function call node op given its outputs' gradients."""
;f_in = [x for x in op.inputs] + out_grads
;f_types = [x.dtype for x in op.inputs]
;f = attr_value_pb2.NameAttrList()
;f.name = op.type
;for k in op.node_def.attr:
;f.attr[k].CopyFrom(op.node_def.attr[k])
;# pylint: disable=protected-access
;in_grads = functional_ops._symbolic_gradient(input=f_in, Tout=f_types, f=f)
;# pylint: enable=protected-access
;return in_grads

; functions annotated with @ops.RegisterGradient(op)
(defmulti build-nodes-gradient (fn [node] (:op node)))

;build entries from sources files like tensorflow/python/ops/math_grad.py
(defmethod build-nodes-gradient "MatMul" [node]
  "something")

;(build-nodes-gradient {:op "MatMul"})
;(ns-unmap *ns* 'build-nodes-gradient)

(defn build-nodes-gradient [nodes target-node]
  )

(defn build-nodes-gradient-descent-optimizer [nodes input-node]
  (let [var-refs (variables/find-variable-nodes nodes)
        gradientable-nodes (find-gradientable-nodes nodes)
        gradients-shape-node (build-node-const nil "DT_INT32" [])
        gradients-const-node (build-node-const 1.0 "DT_FLOAT" [])
        graidents-fill-node (build-node-fill gradients-shape-node gradients-const-node)

        ;;ExpandDims of input-node?!?!?!
        ]
    gradientable-nodes))