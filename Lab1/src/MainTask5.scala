import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.{PolynomialExpansion, VectorAssembler}
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.log4j.Logger
import org.apache.log4j.Level

object MainTask5
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

    // We map each line into an array of features by splitting it at each comma
    val recordsRdd = rdd.map(r => r.split(","))

    // We map each record into a song object
    val songsRdd = recordsRdd.map {
      case Array(year, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12) =>
        Song(year.toDouble, f1.toDouble, f2.toDouble, f3.toDouble)
    }

    // Convert the rdd into a DataFrame
    val songsDf = songsRdd.toDF("Year", "Feature1", "Feature2", "Feature3")
    // Registering the view of the DataFrame
    songsDf.createOrReplaceTempView("songs")

    // TASK 2
    val songsLabel = songsDf.select(songsDf("Year").as("label"),$"Feature1",$"Feature2",$"Feature3")

    val assembler = new VectorAssembler()
      .setInputCols(Array("Feature1", "Feature2", "Feature3"))
      .setOutputCol("raw_features")

    val polynomialExpansion = new PolynomialExpansion().setInputCol("raw_features").setOutputCol("features").setDegree(2)

    val pipelineTwoWay = new Pipeline().setStages(Array(assembler, polynomialExpansion))

    val vectorisedLabeledDF = pipelineTwoWay.fit(songsLabel).transform(songsLabel)
      .drop("Feature1")
      .drop("Feature2")
      .drop("Feature3")
      .drop("raw_feature")

    val trainValidationSplit = new TrainValidationSplit()
      // 80% of the data will be used for training and the remaining 20% for validation.
      .setTrainRatio(0.8)

    val Array(train, test) = vectorisedLabeledDF.randomSplit(Array(0.8,0.2), 0)

    //set the required parameters
    val learningAlgorithm = new LinearRegression()
      .setMaxIter(10)
      .setRegParam(0.1)
      .setElasticNetParam(0.1)

    // set appropriate stages
    val pipeline = new Pipeline().setStages(Array(learningAlgorithm))
    // fit on the training data
    val pipelineModel = pipeline.fit(train)
    // hyper parameter tuning
    // build the parameter grid by setting the values for maxIter and regParam
    val paramGrid = new ParamGridBuilder()
      .addGrid(learningAlgorithm.regParam, Array(0.1, 0.3, 0.6, 0.9))
      .addGrid(learningAlgorithm.maxIter, Array(10, 30, 50))
      .addGrid(learningAlgorithm.elasticNetParam, Array(0.1, 0.3, 0.6, 0.9))
      .build()

    // create the pipeline
    val pipeline_tuning = new Pipeline()
      .setStages(Array(learningAlgorithm))

    val evaluator = new RegressionEvaluator()

    //create the cross validator and set estimator, evaluator, paramGrid
    val cv = new CrossValidator()
      .setEstimator(pipeline_tuning)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(10)

    val cvModel = cv.fit(train)
    val bestModelSummary = cvModel.bestModel.asInstanceOf[PipelineModel].stages(0).asInstanceOf[LinearRegressionModel]

    // Print the coefficients and intercept for linear regression
    println(s"Coefficients: ${bestModelSummary.coefficients} Intercept: ${bestModelSummary.intercept}")
    // print best model RMSE to compare to previous
    println(s"BEST Parameter RMSE: ${bestModelSummary.summary.rootMeanSquaredError}")
    // print best parameters
    println(s"Best Reg: ${bestModelSummary.getRegParam}")
    println(s"Best Elastic: ${bestModelSummary.getElasticNetParam}")
  }


}
