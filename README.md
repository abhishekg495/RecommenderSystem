
# <img src ="https://github.com/abhishekg495/RecommenderSystem/blob/master/Posters/favicon.png?raw=true" width=30px> [Rec-It Ralph](https://share.streamlit.io/abhishekg495/recommendersystem/main.py)

A one-stop shop for movie recommendation, 
integrating multiple recommendation algorithms 
tested and verified on real users for optimised 
results.
## Development Timeline

- Week 1 - Ideation and Initial Documentation
- Week 2 - Backend Development
- Week 3 - Frontend Development
- Week 4 - Final Touches, Integration and Submission


## Tech Stack

**Client:** Streamlit

**Server:** Numpy, Pandas, Scikit-Learn


## Run Locally

Clone the project

```bash
  git clone git@github.com:abhishekg495/RecommenderSystem.git
```

Go to the project directory

```bash
  cd RecommenderSystem
```

Install dependencies

```bash
  pip install -r requirements.txt
```

Start the server

```bash
  streamlit run main.py
```



## Folder Structure

-   `Datasets` - contains all the required data in csv format.

    Webscraping and data processing has been used to produce required information from exisiting datasets on Kaggle.

    -   _average_ratings.csv_
    -   _links.csv_
    -   _movies_features.csv_
    -   _ratings_sorted_movies.csv_
    -   _ratings.csv_

-   `JupyterNotebooks` - contains Python code in Jupyter notebooks format.

    These notebooks are not required for the app to run, but are made available for reference as all the data processing and webscraping was done using Jupyter notebook environment.

    -   BasicRecommender
    -   CollaborativeFiltering
    -   ContentBasedFiltering

-   `Posters` - contains posters for all movies, along with website favicon and other required images.

    Movies' posters have been web scraped from IMDB using Python.

-   `Screenshots` - contains screenshots and demo of the deployed project webpage.

-   `basic_recommender.py` `collaborative_filtering.py` `content_based_filtering.py` - contain Python classes for different recommendation algorithms.

-   `basic_ui.py` `collaborative_ui.py` `content_based_ui.py` - contain Python classes for frontend elements for different recommendation algorithms.

-   `main.py` - driver script to run the application.

-   `posters_printer.py` - contains Python class to render posters for a given list of movies, along with their IMDB lnks.

-   `watchlist.py` - contains Python class to mantain a watchlist of the user's chosen movies.


## Features

- Sort movies based on ratings from thousands of users
- Select your favorite movies to find similar titles
- Search for movies using keywords (actors, tags, directors etc.)
- Get personalised recommendations based on your own ratings
- Automatic light/dark mode toggle based on device theme
- Easy-to-use, smooth and responsive UI
- Cross platform
- Add movies to your watchlist
- Accurate recommendations


## Demo

#### [Click here for the video demo](https://drive.google.com/file/d/12PXyHQ6lLVX45cxmZLl-42jd8pXvV2dh/view?usp=sharing)


## Screenshots

#### Sort by Ratings/Popularity (and choose genres)
![App Screenshot 1](https://github.com/abhishekg495/RecommenderSystem/blob/master/Screenshots/1.png?raw=true)

#### Search for a movie based on its content
![App Screenshot 2](https://github.com/abhishekg495/RecommenderSystem/blob/master/Screenshots/2.png?raw=true)

![App Screenshot 3](https://github.com/abhishekg495/RecommenderSystem/blob/master/Screenshots/3.png?raw=true)

#### Get personalised recommendations based on your own ratings
![App Screenshot 4](https://github.com/abhishekg495/RecommenderSystem/blob/master/Screenshots/4.png?raw=true)

## Lessons Learned

The project started out with a simple inefficient implementation of the ratings/popularity based sorting feature. This taught me how to **sort information stored in a DataFrame** according to different conditions.

The next part was the content-based searching feature. I had to go through a whole bunch of **different metrics and types of NLP models to get the most accurate results** that I could.

The third feature was implemented in the form of personalised recommendations based on a user's ratings for different movies. **Several similarity metrics had to be tried and tested to implement an approach similar to KNN** but in a much more elegant and simplified manner.

Last but not the least, a major part of the development cycle was spent in making the app more user friendly in general and in using several **optimisation techniques like caching** to make the experience smoother for the user.
**Web scraping** also played a major role as gathering accurate data was a much more crucial and tedious task as compared to implementing ML models based on that data.

## Feedback

If you have any feedback, please reach out to me via [LinkedIn](https://www.linkedin.com/in/abhishek-gupta-099288175/)

Looking forward to hearing from you :D
