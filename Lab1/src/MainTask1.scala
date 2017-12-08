import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.sql.SQLContext
import org.apache.log4j.Logger
import org.apache.log4j.Level

object MainTask1 {

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
    import sqlContext._

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
        Song(year.toDouble, f1.toDouble, f2.toDouble, f3.toDouble)
    }

    // Convert the rdd into a DataFrame
    val songsDf = songsRdd.toDF("Year", "Feature1", "Feature2", "Feature3")
    // Registering the view of the DataFrame
    songsDf.createOrReplaceTempView("songs")

    // Counting the number of songs in the dataframe
    val song_number = songsRdd.count()
    val song_number_ = songsDf.count()

    // Printing the result and checking that the two results are equivalent
    assert(song_number == song_number_)
    println("Q1: The dataset contains ".concat(song_number.toString()).concat(" songs."))

    // Counting the number of songs in the period between 1998 and 2000
    val q2 = songsRdd.filter(s => s.year >= 1998 && s.year <= 2000).count()
    val q2_ = songsDf.filter(songsDf("Year") >= 1998 && songsDf("Year") <= 2000).count()

    // Printing the result and checking that the two results are equivalent
    assert(q2 == q2_)
    println("Q2: The dataset contains ".concat(q2_.toString())
      .concat(" songs released between 1998 and 2000."))

    // Counting the number of songs in the period between 1998 and 2000
    val years = songsRdd.map(s => s.year)
    val years_ = songsDf.select("Year")

    val min = years.min()
    val max = years.max()
    val mean = years.reduce(_ + _) / years.count()
    val min_ = sqlContext.sql("SELECT min(Year) FROM songs")
    val max_ = sqlContext.sql("SELECT max(Year) FROM songs")
    val mean_ = sqlContext.sql("SELECT avg(Year) FROM songs")


    println("Q3: Min = ".concat(min.toString())
      .concat("  Max = ").concat(max.toString())
      .concat("  Mean = ").concat(mean.toString()))

    // Counting the number of songs per year between 2000 and 2010
    val songsPerYear = years.filter(x => x>=2000 && x <= 2010)
      .map(x => (x, 1))
      .reduceByKey(_ + _).sortByKey()

    val songsPerYear_ = years_
      .filter(years_("Year") >= 2000 && years_("Year") <= 2010)
      .groupBy("Year").count().sort("Year").show()

    println("Q4: Songs per year in the decade 2000-10")
    songsPerYear.foreach(println)

    println("Some of the questions could have been answered by checking the following table")
    songsDf.describe().show()
  }


}
