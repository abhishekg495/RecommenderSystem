import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


class collaborative_filter:
    def __init__(self):
        self.movies_df = pd.read_csv(
            "Datasets/movies.csv",
            usecols=["movieId", "title"],
            dtype={"movieId": "int32", "title": "str"},
        )
        self.ratings_df = pd.read_csv(
            "Datasets/ratings.csv",
            usecols=["userId", "movieId", "rating"],
            dtype={"userId": "int32", "movieId": "int32", "rating": "float32"},
        )

        ##COMBINING DATASETS
        self.combine_movie_rating = pd.merge(
            self.ratings_df, self.movies_df, on="movieId"
        ).dropna(axis=0, subset=["title"])
        self.movie_ratingCount = (
            self.combine_movie_rating.groupby(by=["title"])["rating"]
            .count()
            .reset_index()
            .rename(columns={"rating": "totalRatingCount"})[
                ["title", "totalRatingCount"]
            ]
        )

        self.rating_with_totalRatingCount = self.combine_movie_rating.merge(
            self.movie_ratingCount, left_on="title", right_on="title", how="left"
        )

        self.popularity_threshold = 50
        self.rating_popular_movie = self.rating_with_totalRatingCount.query(
            "totalRatingCount >= @self.popularity_threshold"
        )
        self.movie_features_df = self.rating_popular_movie.pivot_table(
            index="title", columns="userId", values="rating"
        ).fillna(0)

        self.movie_features_df_matrix = csr_matrix(self.movie_features_df.values)
        self.model_knn = NearestNeighbors(metric="cosine", algorithm="brute")

    def fit_model(self):
        self.model_knn.fit(self.movie_features_df_matrix)

    def get_recommendation(self, query_index):
        distances, indices = self.model_knn.kneighbors(
            self.movie_features_df.iloc[query_index, :].values.reshape(1, -1),
            n_neighbors=10,
        )
        recommendations = []
        for i in range(0, len(distances.flatten())):
            recommendations.append(self.movie_features_df.index[indices.flatten()[i]])

        return recommendations
