import streamlit as st
import numpy as np
from joblib import load, dump

def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances, indexes = model.kneighbors([vectors[movie_index]],n_neighbors=6)
    recommended_movies = [movies.iloc[i].title for i in indexes[0][1:]]
    return recommended_movies

model = load("model/knn_model.joblib")
movies = load("Dataset/preprocessed_data.joblib")
vectors = np.load("vector/vectors.npy")

st.title("Movie Recommender System")

selected_movie_name = st.selectbox(
    'Select a movie',
    movies['title'].values
)

if st.button("Recommend"):
    recommendations = recommend(selected_movie_name)
    for movie in recommendations:
        st.write(movie)