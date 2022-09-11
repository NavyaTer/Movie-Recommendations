from pyspark.ml.recommendation import ALS
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
import pandas as pd

spark = SparkSession.builder \
    .appName('recommendation_colab_als') \
    .config("spark.driver.memory", "5g") \
    .getOrCreate()

spark.conf.set("ml-latest/ratings_modified.csv", "true")
spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
data = spark.read.csv("ml-latest/ratings_modified.csv", inferSchema=True, header=True)
train_data, test_data = data.randomSplit([0.8, 0.2])

als = ALS(maxIter=10, regParam=0.1, nonnegative=True, coldStartStrategy="drop", userCol='userId', itemCol='movieId',
          ratingCol='rating')
model = als.fit(train_data)
predictions = model.transform(test_data)
evaluator = RegressionEvaluator(metricName='rmse', predictionCol='prediction', labelCol='rating')
rmse = evaluator.evaluate(predictions)



def step_1(user_id, content_dataframe=None, content_dataframe_flag=None):
    movies = pd.read_csv('ml-latest/movies.csv', low_memory=False)
    id_to_movie_title = dict(zip(movies['movieId'], movies['title']))

    ratings_modified = pd.read_csv('ml-latest/ratings_modified.csv', low_memory=False)
    ratings_modified_records = ratings_modified.to_dict('records')



    user_suggest = test_data.filter(train_data.userId == user_id).select(['movieId', 'userId'])
    if not content_dataframe_flag:
        user_offer = model.transform(user_suggest)
    else:
        content_dataframe = spark.createDataFrame(content_dataframe)
        user_offer = model.transform(content_dataframe)
    user_offer = user_offer.toPandas()
    user_offer = user_offer.sort_values(by='prediction', ascending=False)
    predicted_rating_for_given_user = dict(zip(user_offer['movieId'], user_offer['prediction']))
    return ratings_modified_records , id_to_movie_title, predicted_rating_for_given_user, rmse

ratings_modified_records , id_to_movie_title, predicted_rating_for_given_user, rmse = step_1(99)

