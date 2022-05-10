import streamlit as st
import pandas as pd

from content_based_filtering import content_based_filter
from basic_recommender import best_average_rating
from collaborative_filtering import collaborative_filter
from update_movies_links import update_movie_links

credits = pd.read_csv("Datasets/tmdb_5000_credits.csv")
credits.rename(columns={"movie_id": "id"}, inplace=True)

if "collaborative_filter" not in st.session_state:
    st.session_state["collaborative_filter"] = collaborative_filter()
    st.session_state["collaborative_filter"].fit_model()

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
st.write(
    best_average_rating(st.session_state["movies_summaries_merged"]).set_index(
        "original_title"
    )
)
st.write(
    """
## Personalised Recommendations
"""
)

movie_name = st.selectbox("Select Movie", credits["title"])

st.write(
    """
### Movies with a similar plot
"""
)
st.write(content_based_filter(movie_name, st.session_state["movies_summaries_merged"]))

st.write(
    """
### Movies with similar ratings from users
"""
)
st.write(st.session_state["collaborative_filter"].get_recommendation(query_index=3))
