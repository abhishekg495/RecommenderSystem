import streamlit as st


class watchlist:
    def __init__(self):
        self.movies_list = list()
        pass

    def add(self, movie_name):
        self.movies_list.append(movie_name)

    def remove(self, movie_name):
        self.movies_list.remove(movie_name)
