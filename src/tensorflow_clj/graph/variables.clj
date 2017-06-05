(ns tensorflow-clj.graph.variables)

(defn build-meta-attr-variable [trainable? variable assign identity]
  {:variable variable
   :assign assign
   :identity identity
   :trainable? trainable?})

(defn find-node-ref [op nodes input-node-name]
  (let [var-nodes (filter #(= op (:op %)) nodes)
        filtered-nodes (filter #(contains? (into #{} (:inputs %)) input-node-name) var-nodes)]
    (first filtered-nodes)))

(defn build-meta-var-ref [nodes trainable? var-node-name]
  (let [identity-node (find-node-ref "Identity" nodes var-node-name)
        assign-node (find-node-ref "Assign" nodes var-node-name)]
    (build-meta-attr-variable trainable? var-node-name (:name assign-node) (:name identity-node))))

(defn find-variable-nodes [nodes]
  (let [var-nodes (filter #(= "VariableV2" (:op %)) nodes)
        var-refs (map #(build-meta-var-ref nodes true (:name %)) var-nodes)]
    var-refs))