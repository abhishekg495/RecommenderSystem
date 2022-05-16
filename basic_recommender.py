import numpy as np
import pandas as pd


class basic_recommender:
    def __init__(self, ratings):
        self.ratings_sorted_movies = pd.read_csv(
            "Datasets/ratings_sorted_movies.csv", index_col=[0]
        )

        self.all_genres_list = []
        for i in range(len(self.ratings_sorted_movies)):
            self.all_genres_list.extend(
                self.ratings_sorted_movies.iloc[i]["genres"].split(" ")
            )
        self.all_genres_list = set(self.all_genres_list)

    def get_columns(self):
        return self.ratings_sorted_movies.columns

    def get_genres(self):
        return list(self.all_genres_list)

    def recommend(self, fav_genres):
        if len(fav_genres) == 0:
            return pd.Series(
                self.ratings_sorted_movies.head(50).set_index("movieId")["title"]
            )
        else:
            self.ratings_sorted_movies[
                "genre count"
            ] = self.ratings_sorted_movies.apply(
                lambda x: len(set(x["genres"].split(" ")).intersection(fav_genres)),
                axis=1,
            )
            recommendations = self.ratings_sorted_movies[
                self.ratings_sorted_movies["genre count"] > 0
            ]
            return pd.Series(recommendations.head(50).set_index("movieId")["title"])
