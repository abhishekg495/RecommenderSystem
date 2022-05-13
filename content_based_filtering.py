import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import sigmoid_kernel
import streamlit as st
import json


class content_based_filter:
    def __init__(self):
        self.credits = pd.read_csv("Datasets/tmdb_5000_credits.csv")
        self.credits.rename(columns={"movie_id": "id"}, inplace=True)
        self.movies = pd.read_csv("Datasets/tmdb_5000_movies.csv")
        self.movies_merged_df = self.movies.merge(self.credits, on="id")

        self.movies_features = self.movies_merged_df[
            ["id", "original_title", "overview", "genres", "cast"]
        ]

        #### CHANGING GENRES STRING TO A LIST OF GENRES ################################################
        self.movies_features.loc[:, "genres"] = pd.Series(
            [
                [j["name"] for j in json.loads(self.movies_features["genres"][i])]
                for i in range(len(self.movies_features))
            ]
        )
        self.movies_features.loc[:, "genres_string"] = pd.Series(
            [
                " ".join([elem for elem in self.movies_features["genres"][i]])
                for i in range(len(self.movies_features))
            ]
        )
        self.movies_features.loc[:, "genres_string"] = self.movies_features[
            "genres_string"
        ].fillna("")
        ###############################################################################################

        ###### EXTRACTING TOP 5 CHARACTERS FROM EACH MOVIE ###############################
        self.movies_features.loc[:, "characters"] = [
            [i["character"] for i in json.loads(self.movies_features["cast"][j])[:5]]
            for j in range(len(self.movies_features))
        ]

        self.movies_features.loc[:, "characters_string"] = [
            ", ".join(
                [character for character in self.movies_features["characters"][j]]
            )
            for j in range(len(self.movies_features))
        ]
        self.movies_features.loc[:, "characters_string"] = self.movies_features[
            "characters_string"
        ].fillna("")
        ##################################################################################

        ##### ADDING UP GENRES, OVERVIEWS AND CHARACTERS (WEIGHTED) ####################
        self.movies_features.loc[:, "overview"] = self.movies_features[
            "overview"
        ].fillna("")
        self.movies_features.loc[:, "genres_overview_characters"] = (
            self.movies_features["genres_string"]
            + 3 * self.movies_features["overview"]
            + 5 * self.movies_features["characters_string"]
        )
        self.movies_features.loc[
            :, "genres_overview_characters"
        ] = self.movies_features["genres_overview_characters"].fillna(" ")
        ################################################################################

        ##### SETTING UP A NEW ENTRY FOR USER INPUT ################################################
        self.movies_features = self.movies_features.append(
            {"original_title": "User Movie", "genres_overview_characters": " "},
            ignore_index=True,
        )
        ###########################################################################################

        self.tfv = TfidfVectorizer(
            min_df=1,
            max_features=None,
            strip_accents="unicode",
            analyzer="word",
            token_pattern=r"\w{1,}",
            ngram_range=(1, 3),
            stop_words="english",
        )

        self.tfv_matrix = self.tfv.fit_transform(
            self.movies_features["genres_overview_characters"]
        )

        self.sig = sigmoid_kernel(self.tfv_matrix, self.tfv_matrix)
        self.indices = pd.Series(
            self.movies_features.index,
            index=self.movies_features["original_title"],
        ).drop_duplicates()

    def get_custom_keywords(self, movie_name="User Movie"):
        return self.movies_features.set_index("original_title").loc[movie_name][
            "genres_overview_characters"
        ]

    def update_model(self, user_given_summary):
        self.movies_features.loc[
            len(self.movies_features) - 1, "genres_overview_characters"
        ] = user_given_summary
        self.tfv_matrix = self.tfv.fit_transform(
            self.movies_features["genres_overview_characters"]
        )
        self.sig = sigmoid_kernel(self.tfv_matrix, self.tfv_matrix)
        return self.movies_features.iloc[-1]["genres_overview_characters"]

    def get_movies_list(self):
        return self.movies_features["original_title"]

    def recommend(self, title="User Movie"):

        # Get the index corresponding to original_title
        self.idx = self.indices[title]

        # Get the pairwsie similarity scores
        self.sig_scores = list(enumerate(self.sig[self.idx]))

        # Sort the movies
        self.sig_scores = sorted(self.sig_scores, key=lambda x: x[1], reverse=True)

        # Scores of the 10 most similar movies
        self.sig_scores = self.sig_scores[1:11]

        # Movie indices
        self.movie_indices = [i[0] for i in self.sig_scores]

        # Top 10 most similar movies
        return self.movies_features["original_title"].iloc[self.movie_indices]
