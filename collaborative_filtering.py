import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


class item_item_collaborative_filter:
    def __init__(self):
        self.movies = pd.read_csv("Datasets/movies.csv")
        self.ratings = pd.read_csv("Datasets/ratings.csv")

        self.ratings = pd.merge(self.movies, self.ratings).drop(
            ["genres", "timestamp"], axis=1
        )

        self.user_ratings = self.ratings.pivot_table(
            index=["title"], columns=["userId"], values="rating"
        )
        self.user_ratings = self.user_ratings.dropna(thresh=10, axis=1).fillna(
            0, axis=1
        )

        self.corr_matrix = self.user_ratings.T.corr(method="pearson")

        self.user_ratings_matrix = csr_matrix(self.user_ratings.values)
        self.model_knn = NearestNeighbors(metric="cosine", algorithm="brute")

    def get_movies_list(self):
        return self.user_ratings.index

    def fit_knn_model(self):
        self.model_knn.fit(self.user_ratings_matrix)

    def knn_recommendation(self, movie_name):
        distances, indices = self.model_knn.kneighbors(
            self.user_ratings.loc[movie_name, :].values.reshape(1, -1),
            n_neighbors=10,
        )
        recommendations = []
        for i in range(0, len(distances.flatten())):
            recommendations.append(self.user_ratings.index[indices.flatten()[i]])

        return recommendations
