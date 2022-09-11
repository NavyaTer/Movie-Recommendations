import pandas as pd
import matplotlib.pyplot as plt

# df_255 = pd.read_csv("ml-latest/255.csv", low_memory=False)
# genres_list = df_255['genres']

movie_id_title_genres = pd.read_csv("ml-latest/movies.csv", low_memory=False)
id_to_movie_genres = dict(zip(movie_id_title_genres['movieId'], movie_id_title_genres['genres']))


#print(id_to_movie_genres)
def make_pie(id_list):
    freq_map = dict()
    for id in id_list:
        #print(id_to_movie_genres[id])

        for genre in id_to_movie_genres[id].split('|'):
            if genre in freq_map:
                print(genre)
                freq_map[genre] += 1
            else:
                freq_map[genre] = 1

    labels = []
    sizes = []
    print(freq_map)
    for x, y in freq_map.items():
        labels.append(x)
        sizes.append(y)

    # Plot
    plt.pie(sizes, labels=labels, autopct='%1.0f%%')
    plt.axis('equal')
    plt.show()
