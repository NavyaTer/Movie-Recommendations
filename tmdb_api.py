import pandas as pd
import requests
import os.path


movies = pd.read_csv('ml-latest/links.csv', low_memory=False)
tmdb_ids = movies['tmdbId'].tolist()



def save_details_json(id):
    details_name = f"./details-overview/{id}.txt"
    flag = os.path.isfile(details_name)
    if not flag:
        with open(details_name, 'w', encoding='utf-8') as file:
            overview_url = f"https://api.themoviedb.org/3/movie/{id}?api_key=<insert TMDB api-key here>&language=en-US"
            try:
                x_json = requests.get(overview_url).json()
                overview = x_json.get('overview', '')
                if overview == '':
                    print("empty overview for ", id)
                file.write(overview)
            except Exception as ex:
                print(ex)


for id in tmdb_ids:
    save_details_json(id)
