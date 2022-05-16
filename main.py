import streamlit as st
import pandas as pd

from content_based_filtering import content_based_filter
from basic_recommender import basic_recommender
from collaborative_filtering import user_collaborative_filter

st.set_page_config(layout="wide")

##################### Genre based filtering cache ####################
def genre_based_rec(genres, weightages):
    return st.session_state["basic_recommender"].recommend(
        genres, weightages[0], weightages[1]
    )


######################################################################

#################### Content based filtering cache ###############################
@st.cache
def update_features_list(features_to_include):
    st.session_state["content_based_recommender"].update_features_combination(
        features_to_include
    )


#################################################################################

####### FUNCTION TO PRINT ALL MOVIES POSTERS IN A GIVEN PANDAS SERIES ##########
def print_movies_posters(recommendations):
    if len(recommendations) == 0:
        st.image(
            "Posters/empty.png", width=300, caption="Try a different search maybe ?"
        )
    else:
        columns = st.columns(5)
        for movie in range(len(recommendations)):
            columns[movie % len(columns)].image(
                "Posters/" + str(recommendations.index[movie]) + ".jpg",
                caption=recommendations.iloc[movie],
            )


#################################################################################


##################### List of algorithms ##################################
recommenders = ["Weighted Averages", "Content-Based", "User Collaborative"]
###########################################################################

# Initalizing datasets ##################################
if "datasets" not in st.session_state:
    st.session_state["datasets"] = {
        "movies": pd.read_csv("Datasets/movies.csv"),
        "ratings": pd.read_csv("Datasets/ratings.csv"),
        "ratings_sorted_movies": pd.read_csv("Datasets/ratings_sorted_movies.csv"),
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
        st.session_state["basic_recommender"] = basic_recommender()
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
    st.sidebar.write(" ")
    with st.sidebar.expander("Sort By"):
        weightage_columns = st.columns(2)
        rating_weightage = int(weightage_columns[0].checkbox("Rating", value=True))
        votes_weightage = int(weightage_columns[1].checkbox("Popularity"))
    genre_recommendations = genre_based_rec(genres, [rating_weightage, votes_weightage])
    print_movies_posters(genre_recommendations)
    # st.write(genre_recommendations)
    # st.write(st.session_state["basic_recommender"].get_columns())

#############################################################################


############## UI For Content Based Filtering ###############################
elif recommender_type == recommenders[1]:

    #### INITIALISE A CONTENT-BASED RECOMMMENDER #############################
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
            st.session_state["movies_list"],
        )
        custom_movie_summary = st.session_state[
            "content_based_recommender"
        ].get_features(custom_movie_titles)

    content_recommendations = st.session_state["content_based_recommender"].recommend(
        features_to_include, custom_movie_summary, strictness=recommendation_strictness
    )
    print_movies_posters(content_recommendations)

##########################################################################


##### UI for Collaborative Filtering #####################################
elif recommender_type == recommenders[2]:

    ###### UI for User-User Collaborative Filtering ###########################
    if "user_collaborative_recommender" not in st.session_state:
        st.session_state["user_collaborative_recommender"] = user_collaborative_filter(
            st.session_state["datasets"]["movies"],
            st.session_state["datasets"]["ratings"],
        )
        st.session_state["user_collaborative_recommender"].fit_knn_model(
            [
                ["101 Dalmatians (One Hundred and One Dalmatians) (1961)", 5],
                ["(500) Days of Summer (2009)", 5],
            ]
        )

    if st.session_state["recommender_type"] != recommenders[2]:
        st.session_state["movies_list"] = st.session_state[
            "user_collaborative_recommender"
        ].get_movies_list()
        st.session_state["recommender_type"] = recommenders[2]

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
    st.write(st.session_state["user_collaborative_recommender"].knn_recommendation())
###################################################################################
