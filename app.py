import streamlit as st
import numpy as np
import requests
from joblib import load, dump

def fetch_poster(movie_id):
    response = requests.get(f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=e0516b6799d52ffcab390a8333980841&language=en-US").json()
    return "https://image.tmdb.org/t/p/original/"+ response['poster_path']


def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances, indexes = model.kneighbors([vectors[movie_index]],n_neighbors=6)
    recommended_movies = []
    posters=[]
    for i in indexes[0][1:]:
        recommended_movies.append(movies.iloc[i].title)
        id = movies.iloc[i].movie_id
        posters.append(fetch_poster(id))
    return recommended_movies,posters

model = load("model/knn_model.joblib")
movies = load("Dataset/preprocessed_data.joblib")
vectors = np.load("vector/vectors.npy")

st.title("Movie Recommender System")

selected_movie_name = st.selectbox(
    'Select a movie',
    movies['title'].values
)

if st.button("Recommend"):
    recommendations, posters = recommend(selected_movie_name)
    for movie, poster in zip(recommendations, posters):
        st.write(movie)
        st.image(poster)