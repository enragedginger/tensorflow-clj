(ns tensorflow-clj.util)

(defn slurp-binary [filename]
  (-> (java.nio.file.FileSystems/getDefault)
      (.getPath "" (into-array String [filename]))
      (java.nio.file.Files/readAllBytes)))

(defn approx= [a b]
  (< (Math/abs (- a b)) 0.0001))

(defn round2
  "Round a double to the given precision (number of significant digits).
  Stolen from http://stackoverflow.com/questions/10751638/clojure-rounding-to-decimal-places"
  [precision d]
  (let [factor (Math/pow 10 precision)]
    (/ (Math/round (* d factor)) factor)))
