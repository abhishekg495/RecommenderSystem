import streamlit as st
import pandas as pd
import numpy as np
from content_based_filtering import content_based_filter
from posters_printer import posters_printer


class content_based_ui:

    ######## INITIALISE REQUIRED DATA AND OBJECTS #################
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
        self.choices = [
            "Find movies similar to your favourite",
            "Use keywords to search for a movie",
        ]
        self.posters_printer = posters_printer()

    ##################################################################

    ##### CACHE DIFFERENT COMBINATIONS OF FEATURES WHILE SORTING ################
    @st.cache
    def update_features_list(features_to_include):
        self.recommender.update_features_combination(features_to_include)

    ##############################################################################

    def render(self):

        #### SELECT WHETHER TO INPUT A MOVIE OR KEYWORDS #####
        content_choice = st.sidebar.radio(
            "What would you like to do",
            self.choices,
        )
        ####################################################

        st.sidebar.write(" ")

        ## SELECT COLUMNS/FEATURES TO MATCH BY ##############################################
        with st.sidebar.expander(
            "Keywords include"
            if content_choice == self.choices[1]
            else "Match movies by:"
        ):
            features_columns = st.columns(2)
            title = features_columns[0].checkbox("Title", value=True)
            synopsis = features_columns[1].checkbox("Synopsis", value=True)
            genres = features_columns[0].checkbox("Genres", value=True)
            actors = features_columns[1].checkbox("Actors", value=True)
            keywords = features_columns[0].checkbox("Tags", value=True)
            directors = features_columns[1].checkbox("Directors", value=True)
        ######################################################################################

        features_to_include = {
            "title": title,
            "synopsis": synopsis,
            "genres": genres,
            "keywords": keywords,
            "actors": actors,
            "directors": directors,
        }

        ###### INPUT USER'S FAVOURITE MOVIES OR KEYWORDS #############################
        if content_choice == self.choices[1]:
            custom_movie_summary = st.text_input("Enter some keywords")
        else:
            custom_movie_titles = st.multiselect(
                "Select your favourite movie(s)",
                self.movies_list,
            )
            custom_movie_summary = self.recommender.get_features(custom_movie_titles)
        ##############################################################################

        ##### GENERATE RECOMMENDATIONS ##############################################
        content_recommendations = self.recommender.recommend(
            features_to_include,
            custom_movie_summary,
        )
        #############################################################################

        self.posters_printer.print(rec=content_recommendations)
