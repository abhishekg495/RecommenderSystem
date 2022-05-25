import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import streamlit as st

IMDB_BASE_URL = "https://www.imdb.com/title/tt"


class collaborative_filter:
    def __init__(self, links):
        self.movies = pd.read_csv("Datasets/movies.csv")
        self.ratings = pd.read_csv("Datasets/ratings.csv")
        self.links = links.set_index("movieId")["imdb_link"]
        self.movieId = self.movies[["movieId", "title"]].set_index(
            "title"
        )  ## GET MOVIE ID USING TITLE
        self.movieTitle = self.movies[["movieId", "title"]].set_index(
            "movieId"
        )  ## GET MOVIE TITLE USING ID

        ## PREPROCESSING DATA TO SET UP A CORRELATION MATRIX BETWEEN MOVIES ##########
        self.ratings = pd.merge(self.movies, self.ratings).drop(
            ["genres", "timestamp"], axis=1
        )

        self.movies_ratings = self.ratings.pivot_table(
            index=["userId"], columns=["movieId"], values="rating"
        )
        self.movies_ratings = self.movies_ratings.dropna(thresh=10, axis=1).fillna(
            0, axis=1
        )
        self.movies_ratings.fillna(0, axis=1, inplace=True)
        self.corr_matrix = self.movies_ratings.corr(method="pearson")
        #############################################################################

    def get_movies_list(self):
        return self.movieTitle.loc[list(self.corr_matrix.index)]["title"]

    def recommend(self, user_ratings):
        if len(user_ratings) == 0:
            return []
        movies_list = []  ## STORE IDs OF MOVIES SELECTED BY USER
        ratings_list = []  ## STORE RATINGS GIVEN BY USER
        for movie, rating in user_ratings:
            movies_list.append(self.movieId.loc[movie]["movieId"])
            ratings_list.append(rating)
        similar_movies = self.corr_matrix[
            movies_list
        ]  ## GET CORRELATION MATRIX FOR SELECTED MOVIES

        ## ADJUST CORRELATION ACCORDING TO USER RATING AND SORT BY CORRELATION #######
        for i in range(len(ratings_list)):
            similar_movies.iloc[
                :, similar_movies.columns.get_loc(movies_list[i])
            ] = similar_movies[movies_list[i]] * (ratings_list[i] - 2.5)
        similar_movies = pd.DataFrame(
            similar_movies.sum(axis=1).sort_values(ascending=False)
        )
        ###############################################################################

        ### ADD A COLUMN FOR TITLES IN SORTED LIST (CONTAINING MOVIE IDs) #############
        similar_movies["title"] = [
            self.movieTitle.loc[movie_id]["title"]
            for movie_id in list(similar_movies.index)
        ]
        ##############################################################################

        ######## ATTACH LINKS AND RETURN #############################################
        return pd.concat(
            [
                similar_movies.head(30)["title"],
                self.links,
            ],
            axis=1,
            join="inner",
        )
        #############################################################################
