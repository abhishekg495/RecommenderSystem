import streamlit as st
import pandas as pd
from content_based_ui import content_based_ui
from basic_ui import basic_recommender_ui
from collaborative_ui import collaborative_ui

st.set_page_config(
    layout="wide", page_title="Rec-It Ralph", page_icon="Posters/favicon.png"
)

style = f"""
<style>
.appview-container .main .block-container{{
        padding-top: 0rem;    }}
footer{{
    visibility: hidden;
}}
</style>"""
st.markdown(style, unsafe_allow_html=True)

##################### List of algorithms ##################################
recommenders = ["Average Ratings", "Content-Based", "User Collaborative"]
###########################################################################

# Initalizing datasets ##################################
if "datasets" not in st.session_state:
    st.session_state["datasets"] = {
        "movies": pd.read_csv("Datasets/movies.csv"),
        "ratings": pd.read_csv("Datasets/ratings.csv"),
        "ratings_sorted_movies": pd.read_csv("Datasets/ratings_sorted_movies.csv"),
        "links": pd.read_csv(
            "Datasets/links.csv",
            index_col=[0],
            dtype={"movieId": int, "imdbId": str, "tmdbId": str, "imdb_link": str},
        ),
    }
#########################################################


st.title("Looking for something to watch ?")
st.sidebar.title("Select a recommendation algorithm")
recommender_type = st.sidebar.selectbox("Choose an algorithm", recommenders)

############## UI For Basic Recommender (Genre Based) ######################
if recommender_type == recommenders[0]:
    if "basic_recommender_ui" not in st.session_state:
        st.session_state["basic_recommender_ui"] = basic_recommender_ui(
            st.session_state["datasets"]["links"]
        )

    st.session_state["basic_recommender_ui"].render()
#############################################################################


############## UI For Content Based Filtering ###############################
elif recommender_type == recommenders[1]:

    #### INITIALISE A CONTENT-BASED UI RENDERING OBECT #############################
    if "content_based_ui" not in st.session_state:
        st.session_state["content_based_ui"] = content_based_ui(
            st.session_state["datasets"]["links"]
        )

    st.session_state["content_based_ui"].render()

###################################################################################


##### UI for Collaborative Filtering #####################################
elif recommender_type == recommenders[2]:

    ###### UI for User-User Collaborative Filtering ###########################
    if "collaborative_ui" not in st.session_state:
        st.session_state["collaborative_ui"] = collaborative_ui(
            st.session_state["datasets"]["links"]
        )

    st.session_state["collaborative_ui"].render()
###################################################################################
