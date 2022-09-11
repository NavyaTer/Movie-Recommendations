import pandas as pd
from pathlib import Path
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


source_dir = Path('details-overview/')
files = source_dir.glob('*.txt')

movie_overview_map = dict()

def process_files(files):
    for file in files:
        with file.open('r') as file_handle :
            for line in file_handle:
                file_name = file_handle.name[17:]
                movie_id = file_name.split('.')[0]
                movie_overview_map[movie_id] = line


process_files(files)
#print(movie_overview_map)
df = pd.DataFrame(list(movie_overview_map.items()))
mapping = {df.columns[0]: 'tmdbId', df.columns[1]: 'overview'}
su = df.rename(columns=mapping)
su.to_csv("custom_tmdb_overview.csv", index=False)
