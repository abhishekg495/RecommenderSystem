import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import sigmoid_kernel
import streamlit as st


class content_based_filter:
    def __init__(self):
        self.movies_summaries = pd.read_csv("Datasets/tmdb_5000_movies.csv")
        self.movies_summaries = self.movies_summaries[["original_title", "overview"]]
        self.movies_summaries = self.movies_summaries.append(
            {"original_title": "User Movie", "overview": ""}, ignore_index=True
        )
        self.tfv = TfidfVectorizer(
            min_df=3,
            max_features=None,
            strip_accents="unicode",
            analyzer="word",
            token_pattern=r"\w{1,}",
            ngram_range=(1, 3),
            stop_words="english",
        )
        self.movies_summaries["overview"] = self.movies_summaries["overview"].fillna("")

        self.tfv_matrix = self.tfv.fit_transform(self.movies_summaries["overview"])

        self.sig = sigmoid_kernel(self.tfv_matrix, self.tfv_matrix)
        self.indices = pd.Series(
            self.movies_summaries.index,
            index=self.movies_summaries["original_title"],
        ).drop_duplicates()

    def get_overview(self, movie_name="User Movie"):
        return self.movies_summaries.set_index("original_title").loc[movie_name][
            "overview"
        ]

    def update_model(self, user_given_summary):
        self.movies_summaries.loc[
            len(self.movies_summaries) - 1, "overview"
        ] = user_given_summary
        self.tfv_matrix = self.tfv.fit_transform(self.movies_summaries["overview"])
        self.sig = sigmoid_kernel(self.tfv_matrix, self.tfv_matrix)
        return self.movies_summaries.iloc[-1]["overview"]

    def get_movies_list(self):
        return self.movies_summaries["original_title"]

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
        return self.movies_summaries["original_title"].iloc[self.movie_indices]
