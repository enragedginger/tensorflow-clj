(ns tensorflow-clj.core
  (:gen-class))

(def ^:dynamic graph nil)

(defmacro with-graph [& body]
  `(binding [graph (org.tensorflow.Graph.)]
     (try
       ~@body
       (finally
         (.close graph)))))

(defn- build-op [op-type op-name attr-map]
  (let [ob (.opBuilder graph op-type (name op-name))]
    (doseq [[attr-name attr-value] attr-map]
      (.setAttr ob attr-name attr-value))
    (-> ob (.build) (.output 0))))

(defn tensor [value]
  (org.tensorflow.Tensor/create value))

(defn constant [name value]
  (let [t (tensor value)]
    (build-op "Const" name {"dtype" (.dataType t) "value" t})))

(defn variable [name]
  (build-op "Variable" name
    {"dtype" org.tensorflow.DataType/DOUBLE
     "shape" (org.tensorflow.Shape/scalar)}))

(defn run-and-fetch [name]
  (with-open [sess (org.tensorflow.Session. graph)]
    (-> sess (.runner)
      (.fetch (name name))
      (.run))))

(defn -main
  "I don't do a whole lot ... yet."
  [& args]
  (println "Hello, World!"))
