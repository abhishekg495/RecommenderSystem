import numpy as np
import pandas as pd


class basic_recommender:
    def __init__(self, movies, ratings):
        self.final_movies = pd.read_csv("Datasets/ratings_sorted_movies.csv")

        self.all_genres_list = []
        for i in range(len(self.final_movies)):
            self.all_genres_list.extend(self.final_movies.iloc[i]["genres"].split(" "))
        self.all_genres_list = set(self.all_genres_list)

    def get_genres(self):
        return list(self.all_genres_list)

    def recommend(self, fav_genres):
        if len(fav_genres) == 0:
            return self.final_movies.head(50)["title"]
        else:
            self.final_movies["genre count"] = self.final_movies.apply(
                lambda x: len(set(x["genres"].split(" ")).intersection(fav_genres)),
                axis=1,
            )
            recommendations = self.final_movies[self.final_movies["genre count"] > 0]
            return recommendations.head(50)["title"].values
