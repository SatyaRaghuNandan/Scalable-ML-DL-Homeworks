import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}
import src.utils.{Array2Vector, Vector2DoubleUDF}

object MainTask5Pipeline
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
      .setOutputCol("raw_features")

    val labelSlicer = new VectorSlicer()
      .setInputCol("vectortokens")
      .setOutputCol("labelVector")

    featureSlicer.setIndices(Array(1,2,3))
    labelSlicer.setIndices(Array(0))

    // val labelScaler = new DoubleUDF(x => x-1922).setInputCol("labelToScale").setOutputCol("label")

    val vectorDisassembler = new Vector2DoubleUDF(x => x.toArray(0))
      .setInputCol("labelVector")
      .setOutputCol("label")

    val polynomialExpansion = new PolynomialExpansion()
      .setInputCol("raw_features")
      .setOutputCol("features")
      .setDegree(2)

    val pipelineDataPreparation = new Pipeline()
      .setStages(Array(regexTokenizer, arr2Vect, labelSlicer, vectorDisassembler, featureSlicer, polynomialExpansion))

    val pipelineModelDataPreparation = pipelineDataPreparation.fit(songsDf)

    val transformedDF = pipelineModelDataPreparation.transform(songsDf)

    val songsDfTransformed = transformedDF
      .drop("Song")
      .drop("raw_data")
      .drop("vectortokens")
      .drop("labelVector")
      .drop("raw_features")


    val trainValidationSplit = new TrainValidationSplit()
      // 80% of the data will be used for training and the remaining 20% for validation.
      .setTrainRatio(0.8)

    val Array(train, test) = songsDfTransformed.randomSplit(Array(0.8,0.2), 0)

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
