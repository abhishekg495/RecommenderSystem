import numpy as np
import pandas as pd
import streamlit as st

IMDB_BASE_URL = "https://www.imdb.com/title/tt"


class basic_recommender:
    def __init__(self, links):
        self.average_ratings = pd.read_csv(
            "Datasets/average_ratings.csv", index_col=[0]
        )
        self.links = links.set_index("movieId")["imdb_link"]

        self.all_genres_list = []
        for i in range(len(self.average_ratings)):
            self.all_genres_list.extend(
                self.average_ratings.iloc[i]["genres"].split(" ")
            )
        self.all_genres_list = set(self.all_genres_list)

    def get_genres(self):
        return list(self.all_genres_list)

    def recommend(self, rating_weightage, votes_weightage):
        self.average_ratings["weighted_average"] = (
            rating_weightage * self.average_ratings["average_rating"]
            + votes_weightage * self.average_ratings["standardized_vote_count"]
        )
        self.average_ratings = self.average_ratings.sort_values(
            "weighted_average", ascending=False
        )

        return pd.concat(
            [
                self.average_ratings.set_index("movieId")[["title", "genres"]],
                self.links,
            ],
            axis=1,
            join="inner",
        )
