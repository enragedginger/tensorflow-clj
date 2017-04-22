(ns tensorflow-clj.core
  (:require [clojure.core.matrix :as matrix])
  (:gen-class))

(def ^:dynamic graph nil)

(defn slurp-binary [filename]
  (-> (java.nio.file.FileSystems/getDefault)
    (.getPath "" (into-array String [filename]))
    (java.nio.file.Files/readAllBytes)))

(defmacro with-graph [& body]
  `(binding [graph (org.tensorflow.Graph.)]
     (try
       ~@body
       (finally
         (.close graph)))))

(defmacro with-graph-file [filename & body]
  `(with-graph
     (.importGraphDef graph (slurp-binary ~filename))
     ~@body))

(defn- build-op [op-type op-name attr-map]
  (let [ob (.opBuilder graph op-type (name op-name))]
    (doseq [[attr-name attr-value] attr-map]
      (.setAttr ob attr-name attr-value))
    (-> ob (.build) (.output 0))))

(defn tensor [value]
  (let [shp (matrix/shape value)]
    (if-not shp
      (org.tensorflow.Tensor/create (float value))
      (org.tensorflow.Tensor/create
        (long-array shp)
        (java.nio.FloatBuffer/wrap
          (float-array (matrix/to-vector value)))))))

(defn tensor->clj [t]
  (assert (instance? org.tensorflow.Tensor t))
  (let [shp (vec (.shape t))]
    (if (empty? shp)
      (.floatValue t)
      (let [buf (java.nio.FloatBuffer/allocate (.numElements t))]
        (.writeTo t buf)
        (matrix/reshape (vec (.array buf))
          shp)))))

(defn constant [name value]
  (let [t (tensor value)]
    (build-op "Const" name {"dtype" (.dataType t) "value" t})))

(defn variable [name]
  (build-op "Variable" name
    {"dtype" org.tensorflow.DataType/FLOAT
     "shape" (org.tensorflow.Shape/scalar)}))

(defn run-graph [feed-ops & fetch-ops]
  (with-open [sess (org.tensorflow.Session. graph)]
    (let [runner (.runner sess)]
      (doseq [[feed-op feed-value] feed-ops]
        (.feed runner (name feed-op) (tensor feed-value)))
      (doseq [fetch-op fetch-ops]
        (.fetch runner (name fetch-op)))
      (vec (map tensor->clj (.run runner))))))

(defn -main
  "I don't do a whole lot ... yet."
  [& args]
  (println "Hello, World!"))
