package src.utils

import org.apache.spark.ml.UnaryTransformer
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.types.DoubleType

class DoubleUDF(override val uid: String, val udf: Double => Double)
    extends UnaryTransformer[Double, Double, DoubleUDF] {

  def this(udf: Double => Double) = this(Identifiable.randomUID("doubleUDF"), udf)

  override protected def createTransformFunc: Double => Double = udf

  override protected def outputDataType: DoubleType = {
    DoubleType
  }
  
  override def copy(extra: ParamMap): DoubleUDF = {
    new DoubleUDF(udf).setInputCol(getInputCol).setOutputCol(getOutputCol)
  }
}