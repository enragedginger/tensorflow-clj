(ns tensorflow-clj.graph.attributes
  (require [tensorflow-clj.util :refer [assoc-not-empty assoc-in-not-empty]]))

(def tf-data-types
  [
   {
    :checker-fn float?
    :val-key :float-val
    :tf-enum-types #{"DT_FLOAT" "DT_DOUBLE" "DT_BFLOAT16"
                     "DT_FLOAT_REF" "DT_DOUBLE_REF" "DT_BFLOAT16_REF"}
    }
   {
    :checker-fn integer?
    :val-key :int-val
    :tf-enum-types #{"DT_INT32" "DT_UINT8" "DT_INT16" "DT_INT8" "DT_INT64" "DT_QINT8" "DT_QUINT8" "DT_QINT32" "DT_QINT16" "DT_QUINT16" "DT_UINT16"
                     "DT_INT32_REF" "DT_UINT8_REF" "DT_INT16_REF" "DT_INT8_REF" "DT_INT64_REF" "DT_QINT8_REF" "DT_QUINT8_REF"
                     "DT_QINT32_REF" "DT_QINT16_REF" "DT_QUINT16_REF" "DT_UINT16_REF"}
    }
   {
    :checker-fn (fn [x] (or (= x true) (= x false)))
    :val-key :bool-val
    :tf-enum-types #{"DT_BOOL" "DT_BOOL_REF"}
    }
   {
    :checker-fn string?
    :val-key :string-val
    :tf-enum-types #{"DT_STRING_REF"}
    }
   ]
  ;DT_COMPLEX64(8), DT_COMPLEX128(18), DT_HALF(19), DT_RESOURCE(20), DT_COMPLEX64_REF(108), DT_COMPLEX128_REF(118),
  ;DT_HALF_REF(119), DT_RESOURCE_REF(120), UNRECOGNIZED(-1)
  )

(defn lookup-by-dtype [dtype]
  (let [matches (filter #(contains? (:tf-enum-types %) dtype) tf-data-types)]
    (first matches)))

(defn find-dtype [attr]
  (or (-> attr :T :type)
      (-> attr :dtype :type)
      (-> attr :value :tensor :dtype)))

(defn build-dims [dims]
  (mapv #(assoc {} :size %) dims))

(defn build-attr [k v]
  ;{:key k :value v}
  {k v})

;;todo should value be "values"? do we need to add support for that?
(defn build-attr-value [value value-dtype dims]
  (let [attr (build-attr :value {
                                 :tensor {
                                          :dtype value-dtype
                                          :tensor_shape {}
                                          }
                                 })
        val-key (-> value-dtype lookup-by-dtype :val-key)]
    (-> attr
        (assoc-in-not-empty [:value :tensor :tensor_shape :dim] (build-dims dims))
        ;;todo add support for nil values?
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