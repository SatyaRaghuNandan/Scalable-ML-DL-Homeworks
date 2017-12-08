import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.feature.RegexTokenizer
import src.utils.Array2Vector
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.feature.VectorSlicer
import src.utils.Vector2DoubleUDF

object MainTask2Pipeline
{
  // dropping info in the console
  Logger.getLogger("org").setLevel(Level.OFF)
  Logger.getLogger("akka").setLevel(Level.OFF)

  case class Song(year: Double, f1: Double, f2: Double, f3: Double)

  def main(args: Array[String]){
    val conf = new SparkConf().setAppName("lab1").setMaster("local")
    val sc = new SparkContext(conf)
    sc.setLogLevel("ERROR")
    val sqlContext = new SQLContext(sc)


    import sqlContext.implicits._

    // Importing the file and parallelisation of the collection

    val filePath = "millionsong.txt"
    val rdd = sc.textFile(filePath)

    val songsDf = rdd.toDF("Song")

    // The delimiter is ',', the features are 13 and of type float
    val regexTokenizer = new RegexTokenizer()

    regexTokenizer.setInputCol("Song")
      .setOutputCol("raw_data")
      .setPattern(",")

    val arr2Vect = new Array2Vector()
      .setInputCol(regexTokenizer.getOutputCol)
      .setOutputCol("vectortokens")

    val featureSlicer = new VectorSlicer()
      .setInputCol("vectortokens")
      .setOutputCol("features")

    val labelSlicer = new VectorSlicer()
      .setInputCol("vectortokens")
      .setOutputCol("labelVector")

    featureSlicer.setIndices(Array(1,2,3))
    labelSlicer.setIndices(Array(0))

    val vectorDisassembler = new Vector2DoubleUDF(x => x.toArray(0))
      .setInputCol("labelVector")
      .setOutputCol("label")

    val pipeline = new Pipeline()
      .setStages(Array(regexTokenizer, arr2Vect, labelSlicer, vectorDisassembler, featureSlicer))

    val pipelineModel = pipeline.fit(songsDf)

    val transformedDF = pipelineModel.transform(songsDf)

    val songsDfTransformed = transformedDF
      .drop("Song")
      .drop("raw_data")
      .drop("vectortokens")
      .drop("labelVector")

    songsDfTransformed.show()

  }


}
