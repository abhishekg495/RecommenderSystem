import numpy as np
import pandas as pd


class basic_recommender:
    def __init__(self):
        self.average_ratings = pd.read_csv(
            "Datasets/average_ratings.csv", index_col=[0]
        )

        self.all_genres_list = []
        for i in range(len(self.average_ratings)):
            self.all_genres_list.extend(
                self.average_ratings.iloc[i]["genres"].split(" ")
            )
        self.all_genres_list = set(self.all_genres_list)

    def get_genres(self):
        return list(self.all_genres_list)

    def recommend(self, fav_genres, rating_weightage, votes_weightage):
        self.average_ratings["weighted_average"] = (
            rating_weightage * self.average_ratings["average_rating"]
            + votes_weightage * self.average_ratings["standardized_vote_count"]
        )
        self.average_ratings = self.average_ratings.sort_values(
            "weighted_average", ascending=False
        )

        ##### RETURN A LIST OF TOP 50 MOVIES IF NO GENRE IS SPECIFIED ##############
        if len(fav_genres) == 0:
            return pd.Series(
                self.average_ratings.head(50).set_index("movieId")["title"]
            )
        ############################################################################

        ### RETURN TOP 50 MOVIES HAVING ATLEAST ONE OF THE CHOSEN GENRES #############
        else:
            self.average_ratings["genre count"] = self.average_ratings.apply(
                lambda x: len(set(x["genres"].split(" ")).intersection(fav_genres)),
                axis=1,
            )
            recommendations = self.average_ratings[
                self.average_ratings["genre count"] > 0
            ]
            return pd.Series(recommendations.head(50).set_index("movieId")["title"])
        ##############################################################################
