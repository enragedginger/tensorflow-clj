(ns tensorflow-clj.core
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
    (let [runner (.runner sess)]
      (print (-> runner
               (.fetch (name name))
               (.run)
               (.get 0)
               (.toString))))))

(defn run-feed-and-fetch [name]
  (with-open [sess (org.tensorflow.Session. graph)]
    (let [runner (.runner sess)]
      (print (-> runner
               (.feed (name name) (tensor 234.0))
               (.fetch (name name))
               (.run)
               (.get 0)
               (.toString))))))

(defn -main
  "I don't do a whole lot ... yet."
  [& args]
  (println "Hello, World!"))
