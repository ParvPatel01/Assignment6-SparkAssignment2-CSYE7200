package edu.neu.coe.csye7200.assign2

import org.apache.spark.sql.{SparkSession, DataFrame}
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.classification.{RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator


object Model extends App {
  val spark = SparkSession.builder()
    .appName("TitanicML")
    .config("spark.master", "local[*]")
    .getOrCreate()

  def loadCSV(filePath: String): DataFrame = {
    spark.read.option("header", "true").option("inferSchema", "true").csv(filePath)
  }

  val trainDF = loadCSV("/Users/parv90/Desktop/Big Data/Assignment2/data/train.csv")
  val testDF = loadCSV("/Users/parv90/Desktop/Big Data/Assignment2/data/test.csv")

  trainDF.printSchema()
  trainDF.show(5)

  // Exploratory Data Analysis
  trainDF.describe().show()

  println("Removing Null Values")
  trainDF.select(trainDF.columns.map(c => sum(when(col(c).isNull, 1).otherwise(0)).alias(c)): _*).show()

  // Feature Engineering
  val df = trainDF
    .withColumn("FamilySize", col("SibSp") + col("Parch") + lit(1))
    .withColumn("IsAlone", when(col("FamilySize") === 1, 1).otherwise(0))
    .drop("PassengerId", "Name", "Ticket", "Cabin")

  val cleanedDF = df.na.fill(Map(
    "Age" -> df.agg(avg("Age")).first().getDouble(0),
    "Embarked" -> "S"
  ))

  val genderIndexer = new StringIndexer().setInputCol("Sex").setOutputCol("SexIndexed").fit(cleanedDF)
  val embarkedIndexer = new StringIndexer().setInputCol("Embarked").setOutputCol("EmbarkedIndexed").fit(cleanedDF)

  val indexedDF = genderIndexer.transform(cleanedDF)
  val finalDF = embarkedIndexer.transform(indexedDF).drop("Sex", "Embarked")

  val featureCols = Array("Pclass", "Age", "SibSp", "Parch", "Fare", "FamilySize", "IsAlone", "SexIndexed", "EmbarkedIndexed")
  val assembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("features")

  val transformedDF = assembler.transform(finalDF).select("features", "Survived")

  // Split training and validation sets
  val Array(trainingData, validationData) = transformedDF.randomSplit(Array(0.8, 0.2), seed = 42)

  // Train the model
  val rf = new RandomForestClassifier()
    .setLabelCol("Survived")
    .setFeaturesCol("features")
    .setNumTrees(100)

  val model = rf.fit(trainingData)

  // Predictions
  val predictions = model.transform(validationData)
  val evaluator = new MulticlassClassificationEvaluator().setLabelCol("Survived").setMetricName("accuracy")

  val accuracy = evaluator.evaluate(predictions)
  println(s"Model Accuracy: ${accuracy * 100}%")

  // Prepare test data for prediction
  val testCleaned = testDF
    .withColumn("FamilySize", col("SibSp") + col("Parch") + lit(1))
    .withColumn("IsAlone", when(col("FamilySize") === 1, 1).otherwise(0))
    .na.fill(Map("Age" -> 30.0, "Fare" -> 35.0, "Embarked" -> "S"))

  val testIndexed = genderIndexer.transform(testCleaned)
  val testFinal = embarkedIndexer.transform(testIndexed).drop("Sex", "Embarked")
  val testTransformed = assembler.transform(testFinal).select("PassengerId", "features")
  val testPredictions = model.transform(testTransformed).select(col("PassengerId"), col("prediction").alias("Survived"))
  testPredictions.write.option("header", "true").csv("/Users/parv90/Desktop/Big Data/Assignment2/data/output.csv")
}
