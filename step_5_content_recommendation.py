import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

#Build Data - dont changes unless needed
# df_custom_tmdb_overview = pd.read_csv("ml-latest/custom_tmdb_overview.csv", low_memory=False)
# #print(df_custom_tmdb_overview)
# df_custom_tmdb_id_title = pd.read_csv("ml-latest/custom_tmdb_id_title.csv", low_memory=False)
# #print(df_custom_tmdb_id_title)
# df_combined = pd.merge(df_custom_tmdb_overview, df_custom_tmdb_id_title, how='inner')
# df_combined.to_csv("ml-latest/custom_combined_tmdb_id__overview_title.csv", index=False)


combined_tmdb_data = pd.read_csv("ml-latest/custom_combined_tmdb_id__overview_title.csv", low_memory=False)
tfidf = TfidfVectorizer(stop_words='english')
combined_tmdb_data['overview'] = combined_tmdb_data['overview'].fillna('')

tfidf_matrix = tfidf.fit_transform(combined_tmdb_data['overview'])

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
indices = pd.Series(combined_tmdb_data.index, index=combined_tmdb_data['title']).drop_duplicates()

def content_recommender(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return combined_tmdb_data['title'].iloc[movie_indices]

print(content_recommender("Dead Poets Society (1989)"))
