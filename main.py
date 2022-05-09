import streamlit as st
import pandas as pd

from content_based_filtering import give_rec
from basic_recommender import best_average_rating

credits = pd.read_csv("Datasets/tmdb_5000_credits.csv")
credits.rename(columns={"movie_id": "id"}, inplace=True)
movies = pd.read_csv("Datasets/tmdb_5000_movies.csv")

movies_merged = movies.merge(credits, on="id")


st.title("Looking for something to watch ?")

st.write(
    """
### What the world is watching
"""
)
st.write(best_average_rating(movies_merged).set_index("original_title"))
st.write(
    """
### Personalised Recommendations
"""
)

movie_name = st.sidebar.selectbox("Select Movie", credits["title"])

st.write(give_rec(movie_name, movies_merged))
