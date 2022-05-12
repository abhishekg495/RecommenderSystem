import streamlit as st
import pandas as pd

from content_based_filtering import content_based_filter
from basic_recommender import basic_recommender
from collaborative_filtering import (
    item_item_collaborative_filter,
    user_user_collaborative_filter,
)

st.title("Looking for something to watch ?")

recommenders = ["Weighted Averages", "Content-Based", "Collaborative"]
collab_algorithms = ["User-User Collaborative", "Item-Item Collaborative"]
recommender_type = st.sidebar.selectbox(
    "Select a recommendation algorithm", recommenders
)

if recommender_type == recommenders[0]:
    st.session_state["recommender"] = basic_recommender()
    genres = st.sidebar.multiselect(
        "Select your genres", st.session_state["recommender"].get_genres()
    )
    st.write(
        """
        ### What the world is watching
    """
    )
    st.write(st.session_state["recommender"].recommend(genres))

elif recommender_type == recommenders[1]:
    st.session_state["recommender"] = content_based_filter()
    st.write(
        """
        ## Personalised Recommendations
    """
    )

    st.write(
        """
        #### Movies with a similar plot
    """
    )
    content_filter_movie_name = st.sidebar.selectbox(
        "Select Movie", st.session_state["recommender"].get_movies_list()
    )
    st.write(st.session_state["recommender"].recommend(content_filter_movie_name))

elif recommender_type == recommenders[2]:

    collab_filter_type = st.sidebar.selectbox(
        "Select the nature of collaborative filter", collab_algorithms
    )

    if collab_filter_type == collab_algorithms[0]:
        st.session_state["recommender"] = user_user_collaborative_filter()
        st.session_state["recommender"].fit_knn_model(
            [
                ["101 Dalmatians (One Hundred and One Dalmatians) (1961)", 5],
                ["(500) Days of Summer (2009)", 5],
            ]
        )
        st.sidebar.write(
            """
        #### Tell us more about your taste
        """
        )
        collab_filter_movie = st.sidebar.selectbox(
            "Select movie",
            st.session_state["recommender"].get_movies_list(),
        )

        st.write(
            """
        ##### Users with a similar choice love these titles
        """
        )
        st.write(st.session_state["recommender"].knn_recommendation())

    elif collab_filter_type == collab_algorithms[1]:
        st.session_state["recommender"] = item_item_collaborative_filter()
        st.session_state["recommender"].fit_knn_model()
        collab_filter_movie = st.sidebar.selectbox(
            "Select movie",
            st.session_state["recommender"].get_movies_list(),
        )

        st.write(
            """
        #### Movies with similar ratings from users
        """
        )

        st.write(
            st.session_state["recommender"].knn_recommendation(collab_filter_movie)
        )

        st.write(
            """
        #### Based on your ratings
        """
        )

        st.write(
            st.session_state["recommender"].corr_recommendation(
                [
                    ("(500) Days of Summer (2009)", 5),
                    ("Alice in Wonderland (2010)", 3),
                    ("Aliens (1986)", 1),
                    ("2001: A Space Odyssey (1968)", 2),
                ]
            )
        )
