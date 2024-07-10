!pip install pyspark

# %%
import pyspark.sql.functions as funcs
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.types import *
from pyspark.sql import SparkSession

# %%
spark = SparkSession.builder\
    .appName("ApacheSparkFinalProject")\
    .master("spark://192.168.56.1:7077")\
    .config("spark.driver.memory","2g")\
    .config("spark.executor.memory", "2g")\
    .getOrCreate()

# %%
'''logger = spark.sparkContext._jvm.org.apache.log4j
logger.LogManager.getLogger("org"). setLevel(logger.Level.ERROR)
logger.LogManager.getLogger("akka").setLevel(logger.Level.ERROR)'''

# %% [markdown]
# # 1. Load Dataset

# %%
iris = spark.read.csv("TrainDf.csv", header=True, inferSchema=True)

# %%
type(iris)

# %%
iris.printSchema()

# %% [markdown]
# # 2. Data Preparation

# %%
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml import Pipeline

# %%
feature_cols = iris.columns[:-1]

# %%
label_indexer = StringIndexer(inputCol="status", outputCol="label")

# %%
assembler = VectorAssembler(inputCols=feature_cols, outputCol='features')

# %%
pipe = Pipeline(stages=[assembler, label_indexer])
pipe_model = pipe.fit(iris)

# %%
data = pipe_model.transform(iris)
data = data.select("features", "label")

# %%
train, test = data.randomSplit([0.70, 0.30])

# %%
test.count()

# %%
train.count()

# %% [markdown]
# # 3. Train Model

# %% [markdown]
# ### 3.1 Decision Tree Algorithm

# %%
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import DecisionTreeClassifier

# %% [markdown]
# #### 3.1.1 Training and Predicting of Model

# %%
dt = DecisionTreeClassifier(labelCol="label", featuresCol="features")
modeldt = dt.fit(train)
predictiondt = modeldt.transform(test)
predictiondt.toPandas().head()

# %% [markdown]
# #### 3.1.2 Confusion Matrix of Decision Tree

# %%
type(predictiondt)

# %%
# Prediction summary
predictiondt.select("prediction", "label")\
    .groupBy("prediction", "label").count()\
    .orderBy("prediction", "label", ascending=True)\
    .withColumn("status", funcs.when(funcs.col("label").isin(1), "Anomaly")\
                .otherwise("Normal")).toPandas().head()

# %%
# Prediction summary
predictiondt.groupBy(["label", "prediction"]).count().toPandas().head()

# %% [markdown]
# #### 3.1.3 Calculation of Accuracy

# %%
evaluatordt = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="accuracy")
dt_accuracy = evaluatordt.evaluate(predictiondt)
print("--- Decision Tree --- ")
print("Accuracy Rate =", round(dt_accuracy, 4))
print("Error Rate =", round((1.0 - dt_accuracy), 4))

# %%
# Convert to RDD for metrics
predictionAndLabel = predictiondt.select("prediction", "label").rdd

from pyspark.mllib.evaluation import MulticlassMetrics
metrics = MulticlassMetrics(predictionAndLabel)
cm = metrics.confusionMatrix()
rows = cm.toArray().tolist()
confusion_matrix = spark.createDataFrame(rows, ["normal", "anomaly"])
confusion_matrix.show()

# %%
# Crosstab for prediction
predictiondt.withColumn("A", funcs.struct("prediction", "label")).crosstab("prediction", "label").show()

# %% [markdown]
# ### 3.2 Random Forest Algorithm

# %%
from pyspark.ml.classification import RandomForestClassifier

# %% [markdown]
# #### 3.2.1 Training and Predicting of Model

# %%
rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=10)
modelrf = rf.fit(train)
predictionrf = modelrf.transform(test)
predictionrf.toPandas().head(3)

# %% [markdown]
# #### 3.2.2 Calculation of Accuracy

# %%
evaluatorrf = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="accuracy")
rf_accuracy = evaluatorrf.evaluate(predictionrf)
print("--- Random Forest Tree --- ")
print("Accuracy Rate =", round(rf_accuracy, 4))
print("Error Rate =", round((1.0 - rf_accuracy), 4))

# %% [markdown]
# #### 3.2.3 Confusion Matrix of Random Forest

# %%
predictionAndLabel_rf = predictionrf.select("prediction", "label").rdd
metrics_rf = MulticlassMetrics(predictionAndLabel_rf)
cm_rf = metrics_rf.confusionMatrix()
rows_rf = cm_rf.toArray().tolist()
confusion_matrix_rf = spark.createDataFrame(rows_rf, ["normal", "anomaly"])
confusion_matrix_rf.show()

# %%
# Crosstab for prediction
predictionrf.withColumn("A", funcs.struct("prediction", "label")).crosstab("prediction", "label").show()

# %%
# Group by prediction and label count
predictionrf.groupBy("label", "prediction").count().show()

# %% [markdown]
# ### 3.3 Naive Bayes Algorithm

# %%
from pyspark.ml.classification import NaiveBayes

# %% [markdown]
# #### 3.3.1 Training and Predicting of Model

# %%
nb = NaiveBayes(smoothing=1.0, modelType="multinomial")
modelnb = nb.fit(train)
predictionnb = modelnb.transform(test)
predictionnb.toPandas().head(3)

# %% [markdown]
# #### 3.3.2 Calculation of Accuracy

# %%
evaluatornb = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="accuracy")
nb_accuracy = evaluatornb.evaluate(predictionnb)
print("--- Naive Bayes --- ")
print("Accuracy Rate =", round(nb_accuracy, 4))
print("Error Rate =", round((1.0 - nb_accuracy), 4))

# %% [markdown]
# ### 3.5 Logistic Regression

# %%
from pyspark.ml.classification import LogisticRegression

# %% [markdown]
# #### 3.5.1 Training and Predicting of Model

# %%
lr = LogisticRegression(regParam=0.01)
modellr = lr.fit(train)
predictionlr = modellr.transform(test)
predictionlr.toPandas().head(3)

# %% [markdown]
# #### 3.5.2 Calculation of Accuracy

# %%
evaluatorlr = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="accuracy")
lr_accuracy = evaluatorlr.evaluate(predictionlr)
print("--- Logistic Regression --- ")
print("Accuracy Rate =", round(lr_accuracy, 4))
print("Error Rate =", round((1.0 - lr_accuracy), 4))

# %% [markdown]
# ## 3.6 Comparison of Accuracy Rate of Algorithms

# %%
print("Decision Tree Accuracy =", round(dt_accuracy, 5))
print("Random Forest Tree Accuracy =", round(rf_accuracy, 5))
print("Logistic Regression Accuracy =", round(lr_accuracy, 5))
print("Naive Bayes Accuracy =", round(nb_accuracy, 5))

# %% [markdown]
# # 4. Streaming Process

# %%
schema = StructType([
    StructField("duration", FloatType(), True),
    StructField("src_bytes", FloatType(), True),
    StructField("dst_bytes", FloatType(), True),
    StructField("land", FloatType(), True),
    StructField("wrong_fragment", FloatType(), True),
    StructField("urgent", FloatType(), True),
    StructField("hot", FloatType(), True),
    StructField("num_failed_logins", FloatType(), True),
    StructField("logged_in", FloatType(), True),
    StructField("num_compromised", FloatType(), True),
    StructField("root_shell", FloatType(), True),
    StructField("su_attempted", FloatType(), True),
    StructField("num_root", FloatType(), True),
    StructField("num_file_creations", FloatType(), True),
    StructField("num_shells", FloatType(), True),
    StructField("num_access_files", FloatType(), True),
    StructField("num_outbound_cmds", FloatType(), True),
    StructField("is_host_login", FloatType(), True),
    StructField("is_guest_login", FloatType(), True),
    StructField("count", FloatType(), True),
    StructField("srv_count", FloatType(), True),
    StructField("serror_rate", FloatType(), True),
    StructField("srv_serror_rate", FloatType(), True),
    StructField("rerror_rate", FloatType(), True),
    StructField("srv_rerror_rate", FloatType(), True),
    StructField("same_srv_rate", FloatType(), True),
    StructField("diff_srv_rate", FloatType(), True),
    StructField("srv_diff_host_rate", FloatType(), True),
    StructField("dst_host_count", FloatType(), True),
    StructField("dst_host_srv_count", FloatType(), True),
    StructField("dst_host_same_srv_rate", FloatType(), True),
    StructField("dst_host_diff_srv_rate", FloatType(), True),
    StructField("dst_host_same_src_port_rate", FloatType(), True),
    StructField("dst_host_srv_diff_host_rate", FloatType(), True),
    StructField("dst_host_serror_rate", FloatType(), True),
    StructField("dst_host_srv_serror_rate", FloatType(), True),
    StructField("dst_host_rerror_rate", FloatType(), True),
    StructField("dst_host_srv_rerror_rate", FloatType(), True),
    StructField("status", StringType(), True)
])

# %% [markdown]
# ### 5.1 Reading Streaming Data

# %%
iris_data = spark.readStream \
    .format("csv") \
    .option("header", True) \
    .option("sep", ",") \
    .schema(schema) \
    .load("data")

# %%
iris.printSchema()

# %%
features_array = iris_data.selectExpr("""
    array(
        CAST(duration AS FLOAT),
        CAST(src_bytes AS FLOAT),
        CAST(dst_bytes AS FLOAT),
        CAST(land AS FLOAT),
        CAST(wrong_fragment AS FLOAT),
        CAST(urgent AS FLOAT),
        CAST(hot AS FLOAT),
        CAST(num_failed_logins AS FLOAT),
        CAST(logged_in AS FLOAT),
        CAST(num_compromised AS FLOAT),
        CAST(root_shell AS FLOAT),
        CAST(su_attempted AS FLOAT),
        CAST(num_root AS FLOAT),
        CAST(num_file_creations AS FLOAT),
        CAST(num_shells AS FLOAT),
        CAST(num_access_files AS FLOAT),
        CAST(num_outbound_cmds AS FLOAT),
        CAST(is_host_login AS FLOAT),
        CAST(is_guest_login AS FLOAT),
        CAST(count AS FLOAT),
        CAST(srv_count AS FLOAT),
        CAST(serror_rate AS FLOAT),
        CAST(srv_serror_rate AS FLOAT),
        CAST(rerror_rate AS FLOAT),
        CAST(srv_rerror_rate AS FLOAT),
        CAST(same_srv_rate AS FLOAT),
        CAST(diff_srv_rate AS FLOAT),
        CAST(srv_diff_host_rate AS FLOAT),
        CAST(dst_host_count AS FLOAT),
        CAST(dst_host_srv_count AS FLOAT),
        CAST(dst_host_same_srv_rate AS FLOAT),
        CAST(dst_host_diff_srv_rate AS FLOAT),
        CAST(dst_host_same_src_port_rate AS FLOAT),
        CAST(dst_host_srv_diff_host_rate AS FLOAT),
        CAST(dst_host_serror_rate AS FLOAT),
        CAST(dst_host_srv_serror_rate AS FLOAT),
        CAST(dst_host_rerror_rate AS FLOAT),
        CAST(dst_host_srv_rerror_rate AS FLOAT)
    ) as arr""",
    "status")

# %% [markdown]
# ### 5.2 Vectorization of streaming data

# %%
tovec_udf = funcs.udf(lambda r: Vectors.dense(r), VectorUDT())
data_stream = features_array.withColumn("features", tovec_udf("arr"))

# %% [markdown]
# # 5. Prediction Process

# %% [markdown]
# ### 5.1 Prediction of Streaming Data

# %%
prediction = modelrf.transform(data_stream)

# %%
type(prediction)

# %%
prediction.printSchema()

# %% [markdown]
# ### 5.2 Adding Sliding Window Time using Current Timestamp

# %%
currentTimeDf = prediction.withColumn("processingTime", funcs.current_timestamp())

# %% [markdown]
# # 6. Start Streaming

# %% [markdown]
# ### 6.1 Option 1 - Using Sliding Windows and Watermarking (Confusion Matrix)

# %%
confusion_matrix = currentTimeDf \
    .withWatermark("processingTime", "5 seconds") \
    .groupBy(funcs.window("processingTime", "3 seconds", "1 seconds"), "status", "prediction") \
    .count() \
    .withColumn("prediction", funcs.when(funcs.col("prediction").isin(1.0), "anomaly") \
                .otherwise("normal")) \
    .orderBy("window")

# %%
q = confusion_matrix.writeStream \
    .outputMode("complete") \
    .format("console") \
    .option("truncate", "false") \
    .start()

# %%
q.awaitTermination()

# %% [markdown]
# ### 6.2 Option 2 - Using Append method

# %%
prediction = prediction.select("features", "status", "prediction")

# %%
q = prediction.writeStream \
    .outputMode("append") \
    .format("console") \
    .start()

# %%
q.awaitTermination()
