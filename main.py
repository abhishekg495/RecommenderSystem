import streamlit as st
import pandas as pd

from content_based_filtering import content_based_filter
from basic_recommender import basic_recommender
from collaborative_filtering import (
    item_item_collaborative_filter,
    user_user_collaborative_filter,
)

st.set_page_config(layout="wide")

##################### Genre based filtering cache ####################
def genre_based_rec(genres):
    return st.session_state["basic_recommender"].recommend(genres)


######################################################################

#################### Content based filtering cache ###############################
@st.cache
def get_movie_summary(movie_name):
    return st.session_state["content_based_recommender"].get_overview(movie_name)


@st.cache
def content_based_rec(movie_name="User Movie"):
    return st.session_state["content_based_recommender"].recommend(movie_name)


@st.cache
def update_content_model(custom_movie_summary):
    st.session_state["content_based_recommender"].update_similarities(
        custom_movie_summary
    )


@st.cache
def update_features_list(features_to_include):
    st.session_state["content_based_recommender"].update_features_combination(
        features_to_include
    )


#################################################################################

####### FUNCTION TO PRINT ALL MOVIES POSTERS IN A GIVEN PANDAS SERIES ##########
def print_movies_posters(recommendations):
    if len(recommendations) == 0:
        st.write("Lookin' kinda empty here. Try another search maybe ?")
    else:
        columns = st.columns(4)
        for movie in range(len(recommendations)):
            columns[movie % 4].image(
                "Posters/" + str(recommendations.index[movie]) + ".jpg",
                caption=recommendations.iloc[movie],
            )


#################################################################################


##################### List of algorithms ##################################
recommenders = ["Weighted Averages", "Content-Based", "Collaborative"]
collab_algorithms = ["User-User Collaborative", "Item-Item Collaborative"]
###########################################################################

# Initalizing datasets ##################################
if "datasets" not in st.session_state:
    st.session_state["datasets"] = {
        "movies": pd.read_csv("Datasets/movies.csv"),
        "ratings": pd.read_csv("Datasets/ratings.csv"),
    }
if "recommender_type" not in st.session_state:
    st.session_state["recommender_type"] = recommenders[0]
#########################################################


st.title("Looking for something to watch ?")

st.sidebar.title("Select a recommendation algorithm")
recommender_type = st.sidebar.selectbox("Choose an algorithm", recommenders)

############## UI For Basic Recommender (Genre Based) ######################
if recommender_type == recommenders[0]:
    if "basic_recommender" not in st.session_state:
        st.session_state["basic_recommender"] = basic_recommender(
            st.session_state["datasets"]["movies"],
            st.session_state["datasets"]["ratings"],
        )
        st.session_state["genres_list"] = st.session_state[
            "basic_recommender"
        ].get_genres()

    if st.session_state["recommender_type"] != recommender_type:
        st.session_state["genres_list"] = st.session_state[
            "basic_recommender"
        ].get_genres()
        st.session_state["recommender_type"] = recommender_type

    genres = st.sidebar.multiselect(
        "Select your genres", st.session_state["genres_list"]
    )
    st.write(
        """
        ### What the world is watching
    """
    )
    genre_recommendations = genre_based_rec(genres)
    print_movies_posters(genre_recommendations)
    # st.write(genre_recommendations)
    # st.write(st.session_state["basic_recommender"].get_columns())

#############################################################################


############## UI For Content Based Filtering ###############################
elif recommender_type == recommenders[1]:
    if "content_based_recommender" not in st.session_state:
        st.session_state["content_based_recommender"] = content_based_filter()
        st.session_state["movies_list"] = st.session_state[
            "content_based_recommender"
        ].get_movies_list()

    if st.session_state["recommender_type"] != recommender_type:
        st.session_state["movies_list"] = st.session_state[
            "content_based_recommender"
        ].get_movies_list()
        st.session_state["recommender_type"] = recommender_type

    content_based_choices = [
        "Find movies similar to your favourite",
        "Use keywords to search for a movie",
    ]
    content_choice = st.sidebar.radio(
        "What would you like to do",
        content_based_choices,
    )
    features_to_include = st.sidebar.multiselect(
        "Keywords include"
        if content_choice == content_based_choices[1]
        else "Match movies by:",
        st.session_state["content_based_recommender"].get_features_list(),
        default=st.session_state["content_based_recommender"].get_features_list(),
    )
    if content_choice == content_based_choices[1]:
        custom_movie_summary = st.text_input("Enter some keywords")
    else:
        custom_movie_titles = st.multiselect(
            "Select your favourite movie(s)",
            st.session_state["content_based_recommender"].get_movies_list(),
        )
        custom_movie_summary = st.session_state[
            "content_based_recommender"
        ].get_features(custom_movie_titles)

    st.write(
        """
        ## Personalised Recommendations
    """
    )

    content_recommendations = st.session_state["content_based_recommender"].recommend(
        features_to_include, custom_movie_summary
    )
    print_movies_posters(content_recommendations)

##########################################################################


##### UI for Collaborative Filtering #####################################
elif recommender_type == recommenders[2]:
    collab_filter_type = st.sidebar.selectbox(
        "Select the nature of collaborative filter", collab_algorithms
    )

    ###### UI for User-User Collaborative Filtering ###########################
    if collab_filter_type == collab_algorithms[0]:
        if "user_collaborative_recommender" not in st.session_state:
            st.session_state[
                "user_collaborative_recommender"
            ] = user_user_collaborative_filter(
                st.session_state["datasets"]["movies"],
                st.session_state["datasets"]["ratings"],
            )
            st.session_state["user_collaborative_recommender"].fit_knn_model(
                [
                    ["101 Dalmatians (One Hundred and One Dalmatians) (1961)", 5],
                    ["(500) Days of Summer (2009)", 5],
                ]
            )

        if st.session_state["recommender_type"] != collab_filter_type:
            st.session_state["movies_list"] = st.session_state[
                "user_collaborative_recommender"
            ].get_movies_list()
            st.session_state["recommender_type"] = collab_filter_type

        st.sidebar.write(
            """
        #### Tell us more about your taste
        """
        )
        collab_filter_movie = st.sidebar.selectbox(
            "Select movie",
            st.session_state["movies_list"],
        )

        st.write(
            """
        ##### Users with a similar choice love these titles
        """
        )
        st.write(
            st.session_state["user_collaborative_recommender"].knn_recommendation()
        )
    #################################################################################

    ##### UI for Item-Item Collaborative Filtering #################################
    elif collab_filter_type == collab_algorithms[1]:
        if "item_collaborative_recommender" not in st.session_state:
            st.session_state[
                "item_collaborative_recommender"
            ] = item_item_collaborative_filter(
                st.session_state["datasets"]["movies"],
                st.session_state["datasets"]["ratings"],
            )
            st.session_state["item_collaborative_recommender"].fit_knn_model()

        if st.session_state["recommender_type"] != collab_filter_type:
            st.session_state["movies_list"] = st.session_state[
                "item_collaborative_recommender"
            ].get_movies_list()
            st.session_state["recommender_type"] = collab_filter_type

        collab_filter_movie = st.sidebar.selectbox(
            "Select movie",
            st.session_state["movies_list"],
        )

        st.write(
            """
        #### Movies with similar ratings from users
        """
        )

        st.write(
            st.session_state["item_collaborative_recommender"].knn_recommendation(
                collab_filter_movie
            )
        )

        st.write(
            """
        #### Based on your ratings
        """
        )

        st.write(
            st.session_state["item_collaborative_recommender"].corr_recommendation(
                [
                    ("(500) Days of Summer (2009)", 5),
                    ("Alice in Wonderland (2010)", 3),
                    ("Aliens (1986)", 1),
                    ("2001: A Space Odyssey (1968)", 2),
                ]
            )
        )
    ###############################################################################
###################################################################################
