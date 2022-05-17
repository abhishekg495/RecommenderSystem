import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import streamlit as st


class collaborative_filter:
    def __init__(self, movies, ratings):
        self.movies = movies
        self.ratings = ratings

        self.id = movies[["movieId", "title"]].set_index("title")

        self.ratings = pd.merge(self.movies, self.ratings).drop(
            ["genres", "timestamp"], axis=1
        )

        self.movies_ratings = self.ratings.pivot_table(
            index=["userId"], columns=["title"], values="rating"
        )
        self.movies_ratings = self.movies_ratings.dropna(thresh=10, axis=1).fillna(
            0, axis=1
        )
        self.movies_ratings.fillna(0, axis=1, inplace=True)
        self.corr_matrix = self.movies_ratings.corr(method="pearson")

    def get_movies_list(self):
        return self.corr_matrix.index

    def recommend(self, user_ratings):
        if len(user_ratings) == 0:
            return []
        movies_list = []
        ratings_list = []
        for movie, rating in user_ratings:
            movies_list.append(movie)
            ratings_list.append(rating)
        similar_movies = self.corr_matrix[movies_list]
        for i in range(len(ratings_list)):
            similar_movies.iloc[
                :, similar_movies.columns.get_loc(movies_list[i])
            ] = similar_movies[movies_list[i]] * (ratings_list[i] - 2.5)
        similar_movies = pd.DataFrame(
            similar_movies.sum(axis=1).sort_values(ascending=False)
        )
        similar_movies["movieId"] = [
            self.id.loc[movie_name]["movieId"] for movie_name in similar_movies.index
        ]
        # similar_movies.sum().sort_values(ascending=False).head(20)
        return pd.Series(
            similar_movies.head(30).reset_index().set_index("movieId")["title"]
        )
