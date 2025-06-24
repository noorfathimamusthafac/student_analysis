import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load and clean dataset
df = pd.read_csv("movies_metadata.csv", low_memory=False)
df = df[df['overview'].notnull() & df['title'].notnull()].reset_index(drop=True)

# Limit the dataset size to speed up processing
df = df.head(5000).copy()

# TF-IDF vectorization
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['overview'])

# Compute cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Create a reverse mapping of movie titles to indices
indices = pd.Series(df.index, index=df['title']).drop_duplicates()

# Recommendation function
def recommend_movies(title, num_recommendations=5):
    title = title.strip()
    if title not in indices:
        return f"Movie '{title}' not found in dataset."
    
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:num_recommendations + 1]
    movie_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[movie_indices].tolist()

# Run recommendation
if __name__ == "__main__":
    movie_title = "The Godfather"
    recommendations = recommend_movies(movie_title)

    print(f"Top 5 movie recommendations for '{movie_title}':")
    if isinstance(recommendations, list):
        for i, movie in enumerate(recommendations, start=1):
            print(f"{i}. {movie}")
    else:
        print(recommendations)
