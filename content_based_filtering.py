import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import sigmoid_kernel


def get_similar_movies(title, movies_cleaned_df):
    tfv = TfidfVectorizer(
        min_df=3,
        max_features=None,
        strip_accents="unicode",
        analyzer="word",
        token_pattern=r"\w{1,}",
        ngram_range=(1, 3),
        stop_words="english",
    )
    movies_cleaned_df["overview"] = movies_cleaned_df["overview"].fillna("")

    tfv_matrix = tfv.fit_transform(movies_cleaned_df["overview"])

    sig = sigmoid_kernel(tfv_matrix, tfv_matrix)
    indices = pd.Series(
        movies_cleaned_df.index, index=movies_cleaned_df["original_title"]
    ).drop_duplicates()

    # Get the index corresponding to original_title
    idx = indices[title]

    # Get the pairwsie similarity scores
    sig_scores = list(enumerate(sig[idx]))

    # Sort the movies
    sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)

    # Scores of the 10 most similar movies
    sig_scores = sig_scores[1:11]

    # Movie indices
    movie_indices = [i[0] for i in sig_scores]

    # Top 10 most similar movies
    return movies_cleaned_df[["original_title", "link"]].iloc[movie_indices]
