from justwatch import JustWatch
import pandas as pd

just_watch = JustWatch(country="US")


def get_movie_link(movie_name):
    try:
        return just_watch.search_for_item(query=movie_name)["items"][0]["offers"][0][
            "urls"
        ]["standard_web"]
    except:
        return "Unavailable"


def update_movie_links(movies):
    movies_updated = movies
    movies_updated["link"] = movies_updated.apply(
        lambda x: get_movie_link(x["original_title"])
        if x["link"] == "Unavailable"
        else x["link"],
        axis=1,
    )
    movies_updated.to_csv("Datasets/final_movies_data.csv")
