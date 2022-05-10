import streamlit as st
import pandas as pd

from content_based_filtering import get_similar_movies
from basic_recommender import best_average_rating
from update_movies_links import update_movie_links

credits = pd.read_csv("Datasets/tmdb_5000_credits.csv")
credits.rename(columns={"movie_id": "id"}, inplace=True)

if "movies_merged" not in st.session_state:
    movies = pd.read_csv("Datasets/final_movies_data.csv")
    st.session_state["movies_merged"] = movies.merge(credits, on="id")

    # update_movie_links(movies_merged)

    # movies = pd.read_csv("Datasets/final_movies_data.csv")
    # movies_merged = movies.merge(credits, on="id")

st.title("Looking for something to watch ?")

st.write(
    """
### What the world is watching
"""
)
st.write(
    best_average_rating(st.session_state["movies_merged"]).set_index("original_title")
)
st.write(
    """
### Personalised Recommendations
"""
)

movie_name = st.sidebar.selectbox("Select Movie", credits["title"])

st.write(get_similar_movies(movie_name, st.session_state["movies_merged"]))
