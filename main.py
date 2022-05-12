import streamlit as st
import pandas as pd

from content_based_filtering import content_based_filter
from basic_recommender import basic_recommender
from collaborative_filtering import (
    item_item_collaborative_filter,
    user_user_collaborative_filter,
)
from update_movies_links import update_movie_links

credits = pd.read_csv("Datasets/tmdb_5000_credits.csv")
credits.rename(columns={"movie_id": "id"}, inplace=True)

if "basic_recommender" not in st.session_state:
    st.session_state["basic_recommender"] = basic_recommender()
if "user_user_collaborative_filter" not in st.session_state:
    st.session_state[
        "user_user_collaborative_filter"
    ] = user_user_collaborative_filter()
    st.session_state["user_user_collaborative_filter"].fit_knn_model(
        [
            ["101 Dalmatians (One Hundred and One Dalmatians) (1961)", 5],
            ["(500) Days of Summer (2009)", 5],
        ]
    )
if "item_item_collaborative_filter" not in st.session_state:
    st.session_state[
        "item_item_collaborative_filter"
    ] = item_item_collaborative_filter()
    st.session_state["item_item_collaborative_filter"].fit_knn_model()

if "movies_summaries_merged" not in st.session_state:
    movies_summaries = pd.read_csv("Datasets/final_movies_data.csv")
    st.session_state["movies_summaries_merged"] = movies_summaries.merge(
        credits, on="id"
    )

    # update_movie_links(movies_merged)

    # movies = pd.read_csv("Datasets/final_movies_data.csv")
    # movies_merged = movies.merge(credits, on="id")

st.title("Looking for something to watch ?")

st.write(
    """
### What the world is watching
"""
)
genres = st.multiselect(
    "Select your genres", st.session_state["basic_recommender"].get_genres()
)
st.write(st.session_state["basic_recommender"].recommend(genres))


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
content_filter_movie_name = st.selectbox("Select Movie", credits["title"])
st.write(
    content_based_filter(
        content_filter_movie_name, st.session_state["movies_summaries_merged"]
    )
)

st.write(
    """
#### Movies with similar ratings from users
"""
)
item_item_collab_filter_movie = st.selectbox(
    "Select movie", st.session_state["item_item_collaborative_filter"].get_movies_list()
)
st.write(
    st.session_state["item_item_collaborative_filter"].knn_recommendation(
        item_item_collab_filter_movie
    )
)


st.write(
    """
### Still not satisfied ?
#### Tell us more about your taste
"""
)
user_user_collab_filter_movie = st.selectbox(
    "Select movie", st.session_state["user_user_collaborative_filter"].get_movies_list()
)

st.write(
    """
##### Users with a similar choice love these titles
"""
)
st.write(st.session_state["user_user_collaborative_filter"].knn_recommendation())

st.write(
    """
#### Based on your ratings
"""
)

st.write(
    st.session_state["item_item_collaborative_filter"].corr_recommendation(
        [
            ("(500) Days of Summer (2009)", 5),
            ("Alice in Wonderland (2010)", 3),
            ("Aliens (1986)", 1),
            ("2001: A Space Odyssey (1968)", 2),
        ]
    )
)
