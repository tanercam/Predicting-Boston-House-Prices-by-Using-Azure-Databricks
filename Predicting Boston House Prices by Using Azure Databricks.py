# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # Predicting Boston House Prices by Using Azure Databricks
# MAGIC 
# MAGIC In this study Boston House Prices were predicted by using Azure Databricks platform. 
# MAGIC The data includes the home values of Boston in 1970's. Five of the 14 diverse variables were used for this small study.
# MAGIC These are:
# MAGIC 
# MAGIC - CRIM - per capita crime rate by town
# MAGIC - RM - average number of rooms per dwelling
# MAGIC - TAX - full-value property-tax rate per $10,000
# MAGIC - LSTAT - % lower status of the population
# MAGIC - MEDV - Median value of owner-occupied homes in $1000's
# MAGIC 
# MAGIC The value of homes (median house values) variable was used as the label and the number of rooms, crime per capita, property tax rate, and percent of the population considered lower class variables were used as the features. Predictions were made by using linear regression. Apache Spark Notebook in Azure Databricks were used for running the codes during the analyses.
# MAGIC 
# MAGIC ###### This small project shows basic machine learning features with Azure Databricks.
# MAGIC ###### The study does not includes some analyses like EDA or Model Tuning etc.!
# MAGIC 
# MAGIC *This is a similar study to Azure Databricks training notebooks. To this study one more feature, new values, and some brief explanations were added.*
# MAGIC 
# MAGIC *Sources: https://docs.microsoft.com,  https://databricks.com, and https://github.com/MicrosoftDocs/mslearn_databricks*

# COMMAND ----------

# MAGIC %run "./Predicting Boston House Prices by Using Azure Databricks"

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Importing Data

# COMMAND ----------

BostonHouseDF = (spark.read
  .option("HEADER", True)
  .option("inferSchema", True)
  .csv("/mnt/training/bostonhousing/bostonhousing/bostonhousing.csv")
)

display(BostonHouseDF)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Creating features column by using VectorAssembler

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler

featureCols = ["rm", "crim", "tax", "lstat"]
assembler = VectorAssembler(inputCols=featureCols, outputCol="features")

BostonHouseFeaturizedDF = assembler.transform(BostonHouseDF)

display(BostonHouseFeaturizedDF)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Model Training (Linear Regression)

# COMMAND ----------

from pyspark.ml.regression import LinearRegression

lr = LinearRegression(labelCol="medv", featuresCol="features")

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Fitting Linear Regression Model to the Data

# COMMAND ----------

lrModel = lr.fit(BostonHouseFeaturizedDF)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### The Coefficients and Intercept of the Model

# COMMAND ----------

print("Coefficients: {0:.1f}, {1:.1f}, {2:.1f}, {3:.1f}".format(*lrModel.coefficients))
print("Intercept: {0:.1f}".format(lrModel.intercept))

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Showing some predictions by creating a sample subDF

# COMMAND ----------

SubsetBostonHouseFeaturizedDF = (BostonHouseFeaturizedDF
  .limit(15)
  .select("features", "medv")
)

display(SubsetBostonHouseFeaturizedDF)

# COMMAND ----------

# MAGIC %md
# MAGIC ###### Displaying Predictions

# COMMAND ----------

PredictionSubsetBostonHouseFeaturizedDF = lrModel.transform(SubsetBostonHouseFeaturizedDF)

display(PredictionSubsetBostonHouseFeaturizedDF)

# COMMAND ----------

# MAGIC %md
# MAGIC Let's predict off a hypothetical data point of a 4 bedroom home with 2.9 crime rate, 222 property tax rate and 13% average lower class.

# COMMAND ----------

from pyspark.ml.linalg import Vectors

data = [(Vectors.dense([4., 2.9, 222, 13.]), )]             
PredictADataPointDF = spark.createDataFrame(data, ["features"])

display(lrModel.transform(PredictADataPointDF))
