(ns tensorflow-clj.graph-ops
  (:require [flatland.protobuf.core :as proto]
            [tensorflow-clj.experimental :as exp]
            [tensorflow-clj.util :as util]
            [camel-snake-kebab.core :as csk])
  (:import
    [org.tensorflow.framework OpList OpList$Builder GraphDef
                              OpDef OpDef$ArgDef OpDef$AttrDef AttrValue ConfigProto DataType AttrValue$ListValue]
    [com.google.protobuf TextFormat]))

(defn build-op-def-map*
  "Get a list of all registered operation definitions,
  like TF_GetAllOpList in the C API.
  Useful for auto generating operations."
  []
  (let [op-list-protobuf-src (slurp "resources/ops.pbtxt")
        op-list-builder (OpList/newBuilder)
        _  (TextFormat/merge ^java.lang.CharSequence op-list-protobuf-src op-list-builder)
        op-list (-> op-list-builder .build .getOpList)
        name-keys (map #(.getName ^OpDef %) op-list)]
    (zipmap name-keys op-list)))

(def op-def-map (memoize build-op-def-map*))

(defn get-op-def
  "Get operation definition from ops.pbtxt"
  [op-name]
  (get (op-def-map) op-name))

(defn keywordize-name
  [name]
  (keyword (csk/->kebab-case name)))

(defn data-type->map
  [^DataType dt-def]
  {
   :name (.name dt-def)
   :number (.getNumber dt-def)
   :name-key (keywordize-name (.name dt-def))
   })

(defn attr-value->map
  [^AttrValue attr-value]
  {:type (data-type->map (.getType attr-value))
   :list (mapv data-type->map (.getTypeList (.getList attr-value)))
   })

(defn attr-def->map
  [^OpDef$AttrDef attr-def]
  {:name (.getName attr-def)
   :description (.getDescription attr-def)
   ;; clj-tf style name
   :name-key (keywordize-name (.getName attr-def))
   :type (.getType attr-def)
   :has-minimum (.getHasMinimum attr-def)
   :minimum (.getMinimum attr-def)
   :allowed-values (attr-value->map (.getAllowedValues attr-def))
   :default-value (attr-value->map (.getDefaultValue attr-def))
   })

(defn arg-def->map
  [^OpDef$ArgDef arg-def]
  {:name (.getName arg-def)
   :description (.getDescription arg-def)
   ;; clj-tf style name
   :keywordized-name (keywordize-name (.getName arg-def))
   :number-attr (.getNumberAttr arg-def)
   ;; TODO :type (.getType arg-def)
   :type (data-type->map (.getType arg-def))
   :type-attr (.getTypeAttr arg-def)
   :type-list-attr (.getTypeListAttr arg-def)
   :type-value (.getTypeValue arg-def)
   :is-ref (.getIsRef arg-def)
   }
  )

(defn op-def->map
  "Get description map of a tensorFlow operation definition."
  [^OpDef op-def]
  {:name (.getName op-def)
   :summary (.getSummary op-def)
   :description (.getDescription op-def)
   :attributes (mapv attr-def->map (.getAttrList op-def))
   :inputs (mapv arg-def->map (.getInputArgList op-def))
   :outputs (mapv arg-def->map (.getOutputArgList op-def))
   })

;(op-def->map (get-op-def "Mul"))
;(op-def->map (get-op-def "ApplyAdagradDA"))
;(op-def->map (get-op-def "SparseApplyAdadelta"))
