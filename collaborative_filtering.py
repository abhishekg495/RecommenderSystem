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

    def single_movie_correlation(self, movie_name, rating):
        similar_ratings = self.corr_matrix[movie_name] * (rating - 2.5)
        similar_ratings = similar_ratings.sort_values(ascending=False)
        return similar_ratings

    def corr_recommendation(self, movie_ratings):
        recommendations = pd.DataFrame()
        for movie, rating in movie_ratings:
            recommendations = recommendations.append(
                self.single_movie_correlation(movie, rating), ignore_index=True
            )
        return recommendations.sum().sort_values(ascending=False).head(20).index


class user_user_collaborative_filter:
    def __init__(self):
        self.movies = pd.read_csv("Datasets/movies.csv")
        self.ratings = pd.read_csv("Datasets/ratings.csv")

        self.ratings = pd.merge(self.movies, self.ratings).drop(
            ["genres", "timestamp"], axis=1
        )

        self.user_ratings = self.ratings.pivot_table(
            index=["userId"], columns=["title"], values="rating"
        )
        self.user_ratings = self.user_ratings.dropna(thresh=10, axis=1).fillna(
            0, axis=1
        )

        self.usable_ratings = self.user_ratings
        self.usable_movies_list = []
        self.user_ratings_matrix = csr_matrix(self.user_ratings.values)

        self.movies_count = len(self.user_ratings.columns)
        self.users_count = len(self.user_ratings.index)

        self.user_ratings = self.user_ratings.append({}, ignore_index=True)
        self.user_ratings.fillna(0, axis=1, inplace=True)

        self.model_knn = NearestNeighbors(metric="cosine", algorithm="brute")

    def get_movies_list(self):
        return self.user_ratings.columns

    def fit_knn_model(self, current_user_ratings):
        self.usable_movies_list = []
        for i in current_user_ratings:
            self.usable_movies_list.append(i[0])
        self.usable_ratings = self.user_ratings[self.usable_movies_list]
        for i in current_user_ratings:
            self.usable_ratings.iloc[self.users_count][i[0]] = i[1]
        self.usable_ratings_matrix = csr_matrix(self.usable_ratings.values)
        self.model_knn.fit(self.usable_ratings_matrix)

    def knn_recommendation(self):
        distances, indices = self.model_knn.kneighbors(
            self.usable_ratings.iloc[self.users_count, :].values.reshape(1, -1),
            n_neighbors=10,
        )
        similar_users = []
        for i in range(0, len(distances.flatten())):
            similar_users.append(self.user_ratings.index[indices.flatten()[i]])

        recommendations = []

        for i in similar_users:
            for j in self.user_ratings.loc[i].sort_values(ascending=False).index[:4]:
                if j not in recommendations:
                    recommendations.append(j)
        return recommendations
