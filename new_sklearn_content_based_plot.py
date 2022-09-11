import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

df_data = pd.read_csv("movielens/movies_metadata.csv", low_memory=False)

#print(df_data)

# array = df_data['vote_count'].values
# array = array.astype(int)
# fifty_percentile_votes = np.percentile(array, 50)
#df_with_votes = df_data.copy(deep=True).loc[df_data['vote_count'] > fifty_percentile_votes]



tfidf = TfidfVectorizer(stop_words='english')
df_data['overview'] = df_data['overview'].fillna('')

tfidf_matrix = tfidf.fit_transform(df_data['overview'])

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
indices = pd.Series(df_data.index, index=df_data['title']).drop_duplicates()


def content_recommender(title, cosine_sim=cosine_sim, df=df_data, indices=indices):

    idx = indices[title]


    sim_scores = list(enumerate(cosine_sim[idx]))


    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:11]


    movie_indices = [i[0] for i in sim_scores]

    return df['title'].iloc[movie_indices]

print(content_recommender('Dead Poets Society'))
