(defproject tensorflow-clj "0.1"
  :description "Gateway from Clojure to Tensorflow"
  :url "https://github.com/enragedginger/tensorflow-clj"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.8.0"]
                 [net.mikera/core.matrix "0.58.0"]
                 [org.tensorflow/tensorflow "1.1.0"]]
  :signing {:gpg-key "enragedginger@gmail.com"}
  :main ^:skip-aot tensorflow-clj.core
  :target-path "target/%s"
  :profiles {:uberjar {:aot :all}})
