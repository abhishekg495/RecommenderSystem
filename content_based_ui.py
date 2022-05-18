import streamlit as st
import pandas as pd
import numpy as np
from content_based_filtering import content_based_filter
from posters_printer import posters_grid


class content_based_ui:
    def __init__(self, links):
        self.recommender = content_based_filter(links)
        self.movies_list = self.recommender.get_movies_list()
        self.features_list = [
            "title",
            "synopsis",
            "genres",
            "keywords",
            "actors",
            "directors",
        ]
        self.posters_printer = posters_grid()

    @st.cache
    def update_features_list(features_to_include):
        self.recommender.update_features_combination(features_to_include)

    def render(self):
        content_based_choices = [
            "Find movies similar to your favourite",
            "Use keywords to search for a movie",
        ]
        content_choice = st.sidebar.radio(
            "What would you like to do",
            content_based_choices,
        )

        st.sidebar.write(" ")
        with st.sidebar.expander(
            "Keywords include"
            if content_choice == content_based_choices[1]
            else "Match movies by:"
        ):
            features_columns = st.columns(2)
            title = features_columns[0].checkbox("Title", value=True)
            synopsis = features_columns[1].checkbox("Synopsis", value=True)
            genres = features_columns[0].checkbox("Genres", value=True)
            actors = features_columns[1].checkbox("Actors", value=True)
            keywords = features_columns[0].checkbox("Tags", value=True)
            directors = features_columns[1].checkbox("Directors", value=True)

        with st.sidebar.expander("Strictness Level"):
            recommendation_strictness = st.slider("", min_value=0, max_value=10)

        features_to_include = {
            "title": title,
            "synopsis": synopsis,
            "genres": genres,
            "keywords": keywords,
            "actors": actors,
            "directors": directors,
        }

        if content_choice == content_based_choices[1]:
            custom_movie_summary = st.text_input("Enter some keywords")
        else:
            custom_movie_titles = st.multiselect(
                "Select your favourite movie(s)",
                self.movies_list,
            )
            custom_movie_summary = self.recommender.get_features(custom_movie_titles)

        content_recommendations = self.recommender.recommend(
            features_to_include,
            custom_movie_summary,
            strictness=recommendation_strictness,
        )
        self.posters_printer.print(rec=content_recommendations)
