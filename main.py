import streamlit as st
import pandas as pd

from content_based_filtering import content_based_filter
from basic_recommender import basic_recommender
from collaborative_filtering import collaborative_filter

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

############# Collaborative Filtering Functions ################################
def add_preference(movie_name, rating):
    st.session_state["preferences"].append([movie_name, rating])


def drop_preference(movie_name, dummy=0):
    for i in range(len(st.session_state["preferences"])):
        if st.session_state["preferences"][i][0] == movie_name:
            del st.session_state["preferences"][i]
            return


####### FUNCTION TO PRINT ALL MOVIES POSTERS IN A GIVEN PANDAS SERIES ##########
def print_movies_posters(recommendations):
    if len(recommendations) == 0:
        st.image("Posters/empty.png", width=300)
    else:
        columns = st.columns(5)
        for movie in range(len(recommendations)):
            try:
                columns[movie % len(columns)].image(
                    "Posters/" + str(recommendations.index[movie]) + ".jpg",
                    caption=recommendations.iloc[movie],
                )
            except:
                columns[movie % len(columns)].image(
                    "Posters/unavailable.png",
                    caption=recommendations.iloc[movie],
                )


#################################################################################


##################### List of algorithms ##################################
recommenders = ["Average Ratings", "Content-Based", "User Collaborative"]
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
    if "collaborative_recommender" not in st.session_state:
        st.session_state["collaborative_recommender"] = collaborative_filter(
            st.session_state["datasets"]["movies"],
            st.session_state["datasets"]["ratings"],
        )
        st.session_state["preferences"] = list()

    if st.session_state["recommender_type"] != recommenders[2]:
        st.session_state["movies_list"] = st.session_state[
            "collaborative_recommender"
        ].get_movies_list()
        st.session_state["preferences"] = list()
        st.session_state["recommender_type"] = recommenders[2]

    st.sidebar.write(
        """
    #### Tell us more about your taste
    """
    )

    movie_name = st.sidebar.selectbox(
        "Select a movie to add", st.session_state["movies_list"]
    )
    rating = st.sidebar.slider("Select a rating", min_value=1, max_value=5)
    delete_movie_name = st.sidebar.select
    submit_movie = st.sidebar.button(
        "Add movie rating", on_click=add_preference, args=(movie_name, rating)
    )

    st.sidebar.write(" ")
    st.sidebar.write(" ")
    if len(st.session_state["preferences"]) > 0:
        delete_movie_name = st.sidebar.selectbox(
            "Want to undo a rating ?",
            pd.DataFrame(st.session_state["preferences"]).rename(
                columns={0: "Title", 1: "Rating"}
            )["Title"],
        )
        remove_rating = st.sidebar.button(
            "Remove", on_click=drop_preference, args=((delete_movie_name, 0))
        )

    if len(st.session_state["preferences"]) > 0:
        st.write(
            pd.DataFrame(st.session_state["preferences"]).rename(
                columns={0: "Title", 1: "Rating"}
            )
        )
    else:
        st.write("#### Try adding some of your own ratings fom the sidebar")
    print_movies_posters(
        st.session_state["collaborative_recommender"].recommend(
            st.session_state["preferences"]
        )
    )
###################################################################################
