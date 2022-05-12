import numpy as np
import pandas as pd


class basic_recommender:
    def __init__(self):
        self.ratings = pd.read_csv("Datasets/ratings.csv")
        self.movies = pd.read_csv("Datasets/movies.csv")
        self.movies["genres"] = self.movies["genres"].str.replace("|", " ")
        self.ratings = self.ratings.merge(self.movies)
        self.ratings.dropna(thresh=10, inplace=True, axis=1)
        self.ratings.fillna(0, axis=1)

        self.grouped_ratings = (
            self.ratings.groupby(by="movieId")["rating"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "average rating", "count": "vote count"})
        )
        self.grouped_ratings = self.grouped_ratings[
            self.grouped_ratings["vote count"] > 10
        ]
        self.grouped_ratings["standardized vote count"] = (
            5 / self.grouped_ratings["vote count"].max()
        ) * self.grouped_ratings["vote count"]
        self.grouped_ratings["weighted average"] = (
            0.5 * self.grouped_ratings["average rating"]
            + 0.5 * self.grouped_ratings["standardized vote count"]
        )
        self.sorted_ratings = (
            self.grouped_ratings["weighted average"]
            .sort_values(ascending=False)
            .head(20)
        )

        self.final_movies = (
            self.movies.set_index("movieId")
            .loc[self.sorted_ratings.index.values, :]
            .merge(self.sorted_ratings, on="movieId")
        )
        self.final_movies["genres list"] = self.final_movies["genres"].str.split(" ")

        self.all_genres_list = []
        for i in range(len(self.final_movies)):
            self.all_genres_list.extend(self.final_movies.iloc[i]["genres list"])
        self.all_genres_list = set(self.all_genres_list)

    def get_genres(self):
        return list(self.all_genres_list)

    def recommend(self, fav_genres):
        if len(fav_genres) == 0:
            return self.final_movies["title"]
        else:
            self.final_movies["genre count"] = self.final_movies.apply(
                lambda x: len(set(x["genres list"]).intersection(fav_genres)), axis=1
            )
            recommendations = self.final_movies[self.final_movies["genre count"] > 0]
            return recommendations.head(20)["title"].values
