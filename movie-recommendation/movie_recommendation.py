'''
The objective of this program is to build a movie recommendation system using the MovieLens 1M dataset.
The dataset contains 1 million ratings from 6000 users on 4000 movies.
The program uses the Alternating Least Squares (ALS) algorithm from the Spark MLlib library to train a collaborative filtering model.
And finally, it generates top 3 movie recommendations for each user in the dataset and saves the results to a CSV file.
'''

# Import libraries
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import explode, col
import pandas as pd

# Initialize Spark Session to run efficiently on laptops
spark = SparkSession.builder \
    .appName("MovieLens1MRecommendation") \
    .master("local[*]") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .config("spark.sql.shuffle.partitions", "8") \
    .getOrCreate()

# Define paths to the .dat files
ratings_path = "./ml-1m/ratings.dat"
movies_path = "./ml-1m/movies.dat"
users_path = "./ml-1m/users.dat"

# Load the ratings data
ratings_df = spark.read \
    .option("delimiter", "::") \
    .option("header", "false") \
    .schema("userId INT, movieId INT, rating FLOAT, timestamp LONG") \
    .csv(ratings_path)

# Load the movies data
movies_df = spark.read \
    .option("delimiter", "::") \
    .option("header", "false") \
    .schema("movieId INT, title STRING, genres STRING") \
    .csv(movies_path)

# Load the users data
users_df = spark.read \
    .option("delimiter", "::") \
    .option("header", "false") \
    .schema("userId INT, gender STRING, age INT, occupation INT, zip STRING") \
    .csv(users_path)

# Drop timestamp from ratings data since it's not needed
ratings_df = ratings_df.drop("timestamp")

# Show data previews
ratings_df.show(5)
movies_df.show(5)
users_df.show(5)

# Split ratings data into train and test sets
train_df, test_df = ratings_df.randomSplit([0.8, 0.2], seed=42)

# Configure ALS model
als = ALS(
    userCol="userId", 
    itemCol="movieId", 
    ratingCol="rating", 
    maxIter=10, 
    regParam=0.1, 
    rank=10, 
    coldStartStrategy="drop"
)

# Train the ALS model
model = als.fit(train_df)

# Generate predictions on test data
predictions = model.transform(test_df)

# Evaluate the model using RMSE
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print("Root Mean Squared Error (RMSE) on test data:", rmse)

# Generate top 3 movie recommendations for each user
user_recommendations = model.recommendForAllUsers(3)

# Expand recommendations and join with movie titles
recommendations = user_recommendations.withColumn("rec_exp", explode("recommendations")) \
    .select("userId", col("rec_exp.movieId"), col("rec_exp.rating"))
recommendations = recommendations.join(movies_df, "movieId").select("userId", "title", "rating")

# Convert to Pandas DataFrame and save as CSV
recommendations_pd = recommendations.toPandas()
recommendations_pd.to_csv("top_3_recommendations_ml1m.csv", index=False)

# Print preview of the saved CSV
print("Top 3 recommendations saved to: top_3_recommendations_ml1m.csv")
print(recommendations_pd.head(10))  # Print the first 10 rows for preview

# Stop Spark Session
spark.stop()