import csv
import os.path
import pickle
import pandas as pd
from pyspark import SparkContext
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import col

RATINGS_CSV = 'data/rating_complete.csv'
RATINGS_USER_CSV = 'data/valoraciones_EP.csv'
usuario = 666666
ruta_modelo_serie = 'modelos/series.model'
ruta_modelo_movie = 'modelos/movies.model'

# Obtener un dataframe con toda la informacion de cada serie partiendo de la recomendacion ALS
def getDFCompleto(df):
    df_completo = df.select('recommendations.anime_id')
    list_series = df_completo.toPandas()['anime_id'][0]
    df_completo = pd.DataFrame(list_series)
    df_completo.columns = ['anime_id']
    df_completo = spark.createDataFrame(df_completo)
    df_completo = df_completo.join(anime, anime.ID == df_completo.anime_id)

    return df_completo



spark = SparkSession.builder.config("spark.driver.memory", "15g").appName("DataFrame").getOrCreate()

with open('data/anime.csv', 'r', encoding='UTF-8') as anime:
    anime_csv = csv.reader(anime, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True)
    df_anime = pd.DataFrame(anime_csv)
    df_anime.columns = df_anime.iloc[0]
    df_anime = df_anime.drop(index=0)


ratings = spark.read.options(delimiter=",", header=True, encoding='UTF-8').csv(RATINGS_CSV)
ratings_user = spark.read.options(delimiter=",", header=False, encoding='UTF-8').csv(RATINGS_USER_CSV)
anime = spark.createDataFrame(df_anime)
anime = anime.withColumn('ID', col('ID').cast(IntegerType()))
anime = anime.drop('rating')
ratings = ratings.union(ratings_user)
ratings = ratings.join(anime, ratings.anime_id == anime.ID)
ratings = ratings.withColumn('anime_id', col('anime_id').cast(IntegerType()))
ratings = ratings.withColumn('user_id', col('user_id').cast(IntegerType()))
ratings = ratings.withColumn('rating', col('rating').cast(IntegerType()))
ratings = ratings.select('user_id', 'anime_id', 'rating', 'English name', 'Japanese name', 'Type')
ratings_movie = ratings.select('*').where(col('Type') == 'Movie')
ratings_serie = ratings.select('*').where(col('Type') != 'Movie')

(training_movie, test_movie) = ratings_movie.randomSplit([0.8, 0.2])
(training_serie, test_serie) = ratings_serie.randomSplit([0.8, 0.2])

# Entrenamos el modelo. La estrategia cold start con 'drop' descarata valores NaN en evaluación
als = ALS(maxIter=5, regParam=0.01, userCol="user_id", itemCol="anime_id", ratingCol="rating", coldStartStrategy="drop")
model_movie = als.fit(training_movie)
model_serie = als.fit(training_serie)

# Evaluamos el modelo con RMSE
predictions_movie = model_movie.transform(test_movie)
predictions_serie = model_serie.transform(test_serie)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
rmse_movie = evaluator.evaluate(predictions_movie)
print("Root-mean-square error movie = " + str(rmse_movie))
rmse_serie = evaluator.evaluate(predictions_serie)
print("Root-mean-square error serie = " + str(rmse_serie))

# Generar las 5 mejores series para el usuario
users = ratings_serie.select('*').where((col('user_id') == usuario))
series_recomendadas = model_serie.recommendForUserSubset(users, 5)

# Generar las 5 mejores peliculas para el usuario
users = ratings_movie.select('*').where((col('user_id') == usuario))
peliculas_recomendadas = model_movie.recommendForUserSubset(users, 5)

df_series = getDFCompleto(series_recomendadas)

with open('series_recomendadas.txt', 'w', encoding='UTF-8') as txtMovies:
    for row in df_series.collect():
        txtMovies.write("Serie Recomendada: " + str(row[0]) + "\n")
        txtMovies.write("Nombre en inglés: " + str(row[2]) + "\n")
        txtMovies.write("Nombre en japonés: " + str(row[5]) + "\n")
        txtMovies.write("Género: " + str(row[1]) + "\n")
        txtMovies.write("Tipo: " + str(row[3]) + "\n")
        txtMovies.write("Emitido: " + str(row[4]) + "\n")
        txtMovies.write("--------------------------------------------------------\n")

df_pelis = getDFCompleto(peliculas_recomendadas)

with open('peliculas_recomendadas.txt', 'w', encoding='UTF-8') as txtMovies:
    for row in df_pelis.collect():
        txtMovies.write("Pelicula Recomendada: " + str(row[0]) + "\n")
        txtMovies.write("Nombre en inglés: " + str(row[2]) + "\n")
        txtMovies.write("Nombre en japonés: " + str(row[5]) + "\n")
        txtMovies.write("Género: " + str(row[1]) + "\n")
        txtMovies.write("Tipo: " + str(row[3]) + "\n")
        txtMovies.write("Emitido: " + str(row[4]) + "\n")
        txtMovies.write("--------------------------------------------------------\n")