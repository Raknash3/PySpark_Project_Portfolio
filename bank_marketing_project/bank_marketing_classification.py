'''
The aim of this project is to predict whether a client will subscribe to a term deposit or not. 
The dataset used in this project is from the UCI Machine Learning Repository. 
The dataset contains information about a marketing campaign of a banking institution. 
The project uses PySpark to build a binary classification model using Logistic Regression and to perfect the model, it uses cross-validation and hyperparameter tuning.
'''

#Import libraries
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline

# Initialize Spark Session
# The spark session is configured to use 4GB of memory and 4 cores.
spark = SparkSession.builder \
    .appName("BankMarketingML") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.cores", "4") \
    .config("spark.sql.shuffle.partitions", "8").getOrCreate()

# Load data
data_path = "bank.csv"  
df = spark.read.csv(data_path, header=True, inferSchema=True)

# List of categorical and numerical columns
categorical_cols = [col for col, dtype in df.dtypes if dtype == "string"]
numerical_cols = [col for col, dtype in df.dtypes if dtype != "string" and col != 'deposit']

# Index and encode categorical columns
indexers = [StringIndexer(inputCol=col, outputCol=col + "_index") for col in categorical_cols]
encoders = [OneHotEncoder(inputCol=col + "_index", outputCol=col + "_vec") for col in categorical_cols]

# Assemble all features into a single dense vector
assembler = VectorAssembler(
    inputCols=[col + "_vec" for col in categorical_cols] + numerical_cols,
    outputCol="features"
)

# Index the label column
label_indexer = StringIndexer(inputCol="deposit", outputCol="label")

# Initialize Logistic Regression
lr = LogisticRegression(labelCol="label", featuresCol="features", maxIter=10)

# Create a Pipeline
pipeline = Pipeline(stages=indexers + encoders + [assembler, label_indexer, lr])

# Split data into training and test sets
train, test = df.randomSplit([0.8, 0.2], seed=12345)

# Set up cross-validation
paramGrid = ParamGridBuilder() \
    .addGrid(lr.regParam, [0.01, 0.1, 1.0]) \
    .build()

#Set up parameter Grid
crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC"),
                          numFolds=5)  # 5-fold cross-validation

# Train the model using cross-validation
cvModel = crossval.fit(train)

# Make predictions on the test set
predictions = cvModel.transform(test)

# Evaluate the model using AUC
evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
auc = evaluator.evaluate(predictions)
print("AUC Score: ", auc)

# Stop Spark Session
spark.stop()
