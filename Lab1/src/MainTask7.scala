import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}
import src.utils.MyLinearModelImpl
import src.utils.MyLinearRegressionImpl

object MainTask7
{
  // dropping info in the console
  Logger.getLogger("org").setLevel(Level.OFF)
  Logger.getLogger("akka").setLevel(Level.OFF)

  // I add the bias column to the song to have in the features
  case class Song(year: Double, f1: Double, f2: Double, f3: Double, b: Double)

  def main(args: Array[String]){
    val conf = new SparkConf().setAppName("lab1").setMaster("local")
    val sc = new SparkContext(conf)
    sc.setLogLevel("ERROR")
    val sqlContext = new SQLContext(sc)


    import sqlContext.implicits._

    // Importing the file and parallelisation of the collection

    val filePath = "millionsong.txt"
    val rdd = sc.textFile(filePath)

    // Print the first 5 rows of
    rdd.take(5).foreach(println)

    // The delimiter is ',', the features are 13 and of type float

    // We map each line into an array of features by splitting it at each comma
    val recordsRdd = rdd.map(r => r.split(","))

    // We map each record into a song object
    val songsRdd = recordsRdd.map {
      case Array(year, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12) =>
        Song(year.toDouble, f1.toDouble, f2.toDouble, f3.toDouble, 1)
    }

    // Convert the rdd into a DataFrame
    val songsDf = songsRdd.toDF("Year", "Feature1", "Feature2", "Feature3", "BiasCol")
    // Registering the view of the DataFrame
    songsDf.createOrReplaceTempView("songs")

    val songsLabel = songsDf.select(songsDf("Year").as("label"),$"Feature1",$"Feature2",$"Feature3",$"BiasCol")

    val assembler = new VectorAssembler()
      .setInputCols(Array("Feature1", "Feature2", "Feature3", "BiasCol"))
      .setOutputCol("features")

    val vectorisedLabeledDF = assembler.transform(songsLabel)
      .drop("Feature1")
      .drop("Feature2")
      .drop("Feature3")
      .drop("BiasCol")

    vectorisedLabeledDF.show(5)

    // TASK 3
    val Array(train, test) = vectorisedLabeledDF.randomSplit(Array(0.8,0.2), 0)

    //set the required parameters
    val learningAlgorithm = new MyLinearRegressionImpl()

    // set appropriate stages
    val pipeline = new Pipeline().setStages(Array(learningAlgorithm))
    // fit on the training data
    val pipelineModel = pipeline.fit(train)
    val myLRModel = pipelineModel.stages(0).asInstanceOf[MyLinearModelImpl]

    println(s"RMSE: ${myLRModel.trainingError(99)}")
    println(s"Weights: ${myLRModel.weights}")

    // make predictions on testing data
    val predictions = pipelineModel.transform(test)
    //print predictions
    predictions.show(5)
  }


}
