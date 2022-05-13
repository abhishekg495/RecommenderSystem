import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import sigmoid_kernel
import streamlit as st


class content_based_filter:
    def __init__(self):
        self.credits = pd.read_csv("Datasets/tmdb_5000_credits.csv")
        self.credits.rename(columns={"movie_id": "id"}, inplace=True)
        self.movies_summaries = pd.read_csv("Datasets/tmdb_5000_movies.csv")
        self.movies_summaries_merged = self.movies_summaries.merge(
            self.credits, on="id"
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
        self.movies_summaries_merged["overview"] = self.movies_summaries_merged[
            "overview"
        ].fillna("")

        self.tfv_matrix = self.tfv.fit_transform(
            self.movies_summaries_merged["overview"]
        )

        self.sig = sigmoid_kernel(self.tfv_matrix, self.tfv_matrix)
        self.indices = pd.Series(
            self.movies_summaries_merged.index,
            index=self.movies_summaries_merged["original_title"],
        ).drop_duplicates()

    def get_overview(self, movie_name):
        return self.movies_summaries_merged.set_index("original_title").loc[movie_name][
            "overview"
        ]

    def get_movies_list(self):
        return self.credits["title"]

    def recommend(self, title):

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
        return self.movies_summaries_merged["original_title"].iloc[self.movie_indices]
