import pandas as pd
import collections
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel



df_custom_tmdb_overview = pd.read_csv("ml-latest/custom_tmdb_overview.csv", low_memory=False)

df_links = pd.read_csv("ml-latest/links.csv", low_memory=False)
tmdb_id_movie_lens = dict(zip(df_links['tmdbId'], df_links['movieId']))
df_movies = pd.read_csv("ml-latest/movies.csv", low_memory=False)
movie_id_title = dict(zip(df_movies['movieId'], df_movies['title']))

tmdb_id_list = list(df_custom_tmdb_overview['tmdbId'])



tmdb_title_map = dict()

for tmdb_id in tmdb_id_list:
    if tmdb_id in tmdb_id_movie_lens:
        movieId = tmdb_id_movie_lens[tmdb_id]
        if movieId in movie_id_title:
            title = movie_id_title[movieId]
            tmdb_title_map[tmdb_id] = title


df = pd.DataFrame(list(tmdb_title_map.items()))
mapping = {df.columns[0]: 'tmdbId', df.columns[1]: 'title'}
su = df.rename(columns=mapping)
su.to_csv("custom_tmdb_id_title.csv", index=False)

