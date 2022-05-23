import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import sigmoid_kernel, cosine_similarity
import streamlit as st


class content_based_filter:
    def __init__(self, links):

        ### READING MOVIES FEATURES DATASET (SCRAPED FROM IMDB) ###############
        self.movies_features = pd.read_csv(
            "Datasets/movies_features.csv", index_col=[0]
        )
        #######################################################################
        self.links = links.set_index("movieId")["imdb_link"]

        self.movies_sorted = pd.read_csv(
            "Datasets/ratings_sorted_movies.csv"
        )  ## READING TOP RATED MOVIES

        ### LIMITING DATASET TO TOP 500 MOVIES TO IMPROVE SPEED ###############
        self.movies_features = self.movies_features.merge(
            self.movies_sorted["movieId"].head(500),
            left_index=True,
            right_on="movieId",
            how="right",
        )
        ######################################################################

        if ("movieId") in list(self.movies_features.columns):
            self.movies_features.set_index("movieId", inplace=True)
        self.movies_features = self.movies_features.fillna(" ")
        self.features_list = [
            "title",
            "synopsis",
            "genres",
            "keywords",
            "actors",
            "directors",
        ]

        ########### INITIALISING TF-IDF MODEL #########################################
        self.tfv = TfidfVectorizer(
            min_df=1,
            max_features=None,
            strip_accents="unicode",
            analyzer="word",
            token_pattern=r"\w{1,}",
            ngram_range=(1, 3),
            stop_words="english",
        )
        self.tfv.fit(self.movies_features["features"])
        self.movies_features["similarity"] = self.movies_features["features"].apply(
            lambda x: self.get_similarity(x, " ")
        )
        #################################################################################

    ######## COMPARE TWO SENTENCES FOR SIMILARITY ##############################
    def get_similarity(self, sentence1, sentence2):
        return cosine_similarity(
            self.tfv.transform([sentence1]), self.tfv.transform([sentence2])
        )[0][0]

    ###########################################################################

    ###### GET A LIST OF THE FEATURES AVAILABLE FOR EACH MOVIE ###############
    def get_features_list(self):
        return self.features_list

    #########################################################################

    ##### COMPUTE FINAL FEATURES STRING BASED ON FEATURES SELECTED BY THE USER #####
    def update_features_combination(self, features_to_include):
        self.movies_features = self.movies_features.fillna(" ")
        for i in range(len(self.movies_features) - 1):
            self.movies_features.iat[
                i, self.movies_features.columns.get_loc("features")
            ] = " "
            for j in features_to_include:
                if features_to_include[j]:
                    self.movies_features.iat[
                        i, self.movies_features.columns.get_loc("features")
                    ] += (
                        self.movies_features.iat[
                            i,
                            self.movies_features.columns.get_loc(j),
                        ]
                        + " "
                    )
        self.tfv.fit(self.movies_features["features"])

    ###################################################################################

    ###### COMPUTE SIMILARITY OF EACH MOVIE'S FEATURES WITH A GIVEN STRING ##############
    def update_similarities(self, user_given_summaries):
        self.movies_features["similarity"] = self.movies_features["features"].apply(
            lambda x: sum(
                [(self.get_similarity(x, summary)) for summary in user_given_summaries]
            )
            if isinstance(user_given_summaries, list)
            else (self.get_similarity(x, user_given_summaries))
        )

    #####################################################################################

    ######## GET A LIST OF ALL AVAILABLE TITLES #########################################
    def get_movies_list(self):
        return self.movies_features["title"]

    #####################################################################################

    #### EXTRACT THE FEATURES OF ALL MOVIES SELETED BY THE USER IN A LIST ################
    def get_features(self, movies):
        features_list = []
        chosen_movies_data = self.movies_features[
            list(title in movies for title in self.movies_features["title"])
        ]
        return list(chosen_movies_data["features"])

    ######################################################################################

    ###### RECOMMEND MOVIES BASED ON FEATURE SET, SELECTED MOVIES/KEYWORDS AND STRICTNESS LEVEL #########
    def recommend(self, features_to_include, user_given_summary):

        self.update_features_combination(features_to_include)
        self.update_similarities(user_given_summary)

        similarity_scores = self.movies_features[["similarity", "title"]].sort_values(
            "similarity", ascending=False
        )
        return pd.concat(
            [
                similarity_scores[similarity_scores["similarity"] > 0.01]["title"],
                self.links,
            ],
            axis=1,
            join="inner",
        )

    ####################################################################################################
