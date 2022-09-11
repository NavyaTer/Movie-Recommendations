from step_2_frequency import return_map
from step_5_content_recommendation import content_recommender
from step_1_collaborative import step_1
import pandas as pd


map = return_map()

movies = pd.read_csv('ml-latest/movies.csv', low_memory=False)
movie_title_to_id_map = dict(zip(movies['title'], movies['movieId']))
movie_id_to_title_map = dict(zip(movies['movieId'], movies['title']))


title_list = []

for title, count in map.items():
    title_list.append(title)

su_list = []

for title in title_list:
    print(f"======={title}=========")
    recommendations = content_recommender(title)
    recommendations_ids_map = {}
    for recommendation in recommendations:
        recommendations_ids_map[movie_title_to_id_map[recommendation]] = 99
        print(f"{recommendation} , {movie_title_to_id_map[recommendation]}")
    df = pd.DataFrame(list(recommendations_ids_map.items()))
    mapping = {df.columns[0]: 'movieId', df.columns[1]: 'userId'}
    su = df.rename(columns=mapping)
    su_list.append(su)


rating_list = []
movie_id_list = []
for su in su_list:
    ratings_modified_records, id_to_movie_title, predicted_rating_for_given_user, rmse = step_1(99, content_dataframe=su, content_dataframe_flag=True)
    for movie_id, rating in predicted_rating_for_given_user.items():
            rating_list.append((round(rating,5)))
            movie_id_list.append(movie_id)

rating_list = set(rating_list)
movie_id_list = set(movie_id_list)
print(rating_list)
print(len(rating_list))
print(movie_id_list)
print(len(movie_id_list))
