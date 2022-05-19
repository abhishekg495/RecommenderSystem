import streamlit as st
import pandas as pd
import numpy as np
from posters_printer import posters_grid
from collaborative_filtering import collaborative_filter


class collaborative_ui:
    def __init__(self, links_data):
        self.recommender = collaborative_filter(links_data)
        self.user_ratings = list()
        self.movies_list = self.recommender.get_movies_list()
        self.posters_printer = posters_grid()

    def add_preference(self, movie_name, rating):
        self.user_ratings.append([movie_name, rating])

    def drop_preference(self, movie_name, dummy=0):
        for i in range(len(self.user_ratings)):
            if self.user_ratings[i][0] == movie_name:
                del self.user_ratings[i]
                return

    def render(self):
        st.sidebar.write(
            """
        #### Tell us more about your taste
        """
        )

        movie_name = st.sidebar.selectbox("Select a movie to add", self.movies_list)
        rating = st.sidebar.slider("Select a rating", min_value=1, max_value=5)
        submit_movie = st.sidebar.button(
            "Add movie rating", on_click=self.add_preference, args=(movie_name, rating)
        )

        st.sidebar.write(" ")
        st.sidebar.write(" ")
        if len(self.user_ratings) > 0:
            delete_movie_name = st.sidebar.selectbox(
                "Want to undo a rating ?",
                pd.DataFrame(self.user_ratings).rename(
                    columns={0: "Title", 1: "Rating"}
                )["Title"],
            )
            remove_rating = st.sidebar.button(
                "Remove", on_click=self.drop_preference, args=((delete_movie_name, 0))
            )

        if len(self.user_ratings) > 0:
            with st.expander("Your ratings"):
                st.write(
                    pd.DataFrame(self.user_ratings)
                    .rename(columns={0: "Title", 1: "Rating"})
                    .set_index("Title")
                )
        else:
            st.write("#### Try adding some of your own ratings fom the sidebar")

        self.posters_printer.print(rec=self.recommender.recommend(self.user_ratings))
