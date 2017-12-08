package se.kth.spark.lab1.task6
import org.apache.spark.ml.linalg.{Matrices, SparseVector, Vector, Vectors}
import org.apache.spark.hacks.VectorType
import src.utils.VectorHelper
import org.apache.spark._
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.{DataFrame, Row, SQLContext}
import org.apache.spark.ml.PipelineModel

object Main {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("lab1").setMaster("local")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    val v = scala.collection.immutable.Vector.empty
    println(v)

    // Add elements from List of Ints to end of vector.
    val v2 = Vectors.dense(Array(15.0,2.0,34.0))
    val v1 = Vectors.dense(Array(0.0,8.0,23.0))



    import sqlContext.implicits._
    import sqlContext._
/*
    val filePath = "src/main/resources/millionsong.txt"
    val obsDF: DataFrame = ???

    val myLR = ???
    val lrStage = ???
    val pipelineModel: PipelineModel = ???
    val myLRModel = pipelineModel.stages(lrStage).asInstanceOf[MyLinearModelImpl]

    //print rmse of our model
    //do prediction - print first k

    */
  }
}