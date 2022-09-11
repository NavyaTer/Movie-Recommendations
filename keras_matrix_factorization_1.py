import matplotlib.pyplot as plt
from math import sqrt
import pandas as pd
from keras.models import Model, Sequential
from keras.layers import Embedding, Flatten, Input, Concatenate, dot, Dropout, Dense
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import mean_absolute_error



dataset = pd.read_csv('../ml-latest/ratings_modified.csv', low_memory=False)

movieId_to_new_id = dict()
new_id_to_movie_Id = dict()
id = 1
for index, row in dataset.iterrows():
    if movieId_to_new_id.get(row['movieId']) is None:
        movieId_to_new_id[row['movieId']] = id
        new_id_to_movie_Id[id] = row['movieId']
        dataset.at[index, 'movieId'] = id
        id += 1
    else:
        dataset.at[index, 'movieId'] = movieId_to_new_id.get(row['movieId'])

num_users = len(dataset.userId.unique())
num_movies = len(dataset.movieId.unique())

train, test = train_test_split(dataset, test_size=0.2)


latent_dim = 20

movie_input = Input(shape=[1],name='movie-input')
movie_embedding = Embedding(num_movies + 1, latent_dim, name='movie-embedding')(movie_input)
movie_vec = Flatten(name='movie-flatten')(movie_embedding)

user_input = Input(shape=[1],name='user-input')
user_embedding = Embedding(num_users + 1, latent_dim, name='user-embedding')(user_input)
user_vec = Flatten(name='user-flatten')(user_embedding)

merged_vectors = dot([user_vec, movie_vec], name='Dot_Product', axes=1)

dense_1 = Dense(16, name='dense-1', activation='relu')(merged_vectors)
drop_1 = Dropout(0.2, name='fc-1-dropout')(dense_1)
output = Dense(1, name='fc-2', activation='relu')(drop_1)
model = Model([user_input, movie_input], output)
model.compile('adam', 'mean_squared_error')
model.summary()
epoch_count = 10
history= model.fit([train.userId, train.movieId], train.rating, epochs=epoch_count)

movie_id_list = list(set(test.movieId.tolist()))
user_id_99 = [99] * len(movie_id_list)
user_id_99_df = pd.DataFrame({'userId': user_id_99})
movie_id_list_df = pd.DataFrame({'movieId': movie_id_list})





y_hat = np.round(model.predict([test.userId, test.movieId]), decimals=2)
y_true = test.rating
print(sqrt(mean_absolute_error(y_true, y_hat)))


prediction_user_99 = np.round(model.predict([user_id_99_df.userId, movie_id_list_df.movieId]), decimals=5)
prediction_user_99_list = prediction_user_99.tolist()
prediction_mappings = dict(zip(movie_id_list, prediction_user_99_list))
prediction_mappings = {k: v for k, v in sorted(prediction_mappings.items(), key=lambda item: item[1], reverse=True)}

df_movies = pd.read_csv("../ml-latest/movies.csv", low_memory=False)
movie_id_title = dict(zip(df_movies['movieId'], df_movies['title']))
top_movies = list(prediction_mappings.items())[:10]

for top_movie in top_movies:
    movieId = f"{top_movie[0]}"
    rating = f"{top_movie[1]}"
    title = movie_id_title[new_id_to_movie_Id[int(movieId)]]
    row = f"{title}, {rating}, {movieId}"
    print(row)



loss = history.history['loss']
plt.figure(figsize = (12,10))
plt.plot(loss)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()


#https://calvinfeng.gitbook.io/machine-learning-notebook/supervised-learning/recommender/neural_collaborative_filtering
