package src.utils

import org.apache.spark.ml.linalg.{Matrices, Vector, Vectors}

object VectorHelper {
  def dot(v1: Vector, v2: Vector): Double = {
    val a1 = v1.toArray
    val a2 = v2.toArray
    val result = a1.zip(a2).map{
      case (x,y) =>
        x * y
    }.sum

    return result
  }

  def dot(v: Vector, s: Double): Vector = {
    val result = v.toArray.map(a => a*s)
    return Vectors.dense(result)
  }

  def sum(v1: Vector, v2: Vector): Vector = {
    val a1 = v1.toArray
    val a2 = v2.toArray
    val result = a1.zip(a2).
      map{
      case (x,y) =>
        x + y
    }

    return Vectors.dense(result)
  }

  def fill(size: Int, fillVal: Double): Vector = {
    val vectorArray = Array.fill(size){fillVal}

    return Vectors.dense(vectorArray)
  }
}