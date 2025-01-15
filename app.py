import os
import streamlit as st
import numpy as np
import requests
from joblib import load
import gdown

# Define file paths
model_path = "model/knn_model.joblib"
data_path = "Dataset/preprocessed_data.joblib"
vectors_path = "vector/vectors.npy"

# Create directories if they don't exist
os.makedirs("model", exist_ok=True)
os.makedirs("vector", exist_ok=True)

# Cache the file download process
@st.cache_data
def download_files():
    try:
        # Download model file if it doesn't exist
        if not os.path.exists(model_path):
            model_url = "https://drive.google.com/uc?id=1T-CkSrKoZ76pfbakybT9KbSHarhlD7MS"
            gdown.download(model_url, model_path, quiet=False)

        # Download vectors file if it doesn't exist
        if not os.path.exists(vectors_path):
            vectors_url = "https://drive.google.com/uc?id=1DuFnAz12v_tLs_cxPqRy-lsXvT0rBv4e"
            gdown.download(vectors_url, vectors_path, quiet=False)
    except Exception as e:
        st.error(f"Error downloading files: {e}")

# Cache the model and data loading
@st.cache_resource
def load_model():
    return load(model_path)

@st.cache_resource
def load_data():
    return load(data_path)

@st.cache_resource
def load_vectors():
    return np.load(vectors_path)

# Download files (if not already downloaded)
download_files()

# Load the model, data, and vectors
model = load_model()
movies = load_data()
vectors = load_vectors()



def fetch_poster(movie_id):
    """Fetch the movies posters from the TMDB API and return the poster path"""
    try:
        response = requests.get(f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=e0516b6799d52ffcab390a8333980841&language=en-US")
        response.raise_for_status()  #Raise an error for bad response
        poster_path = response.json().get('poster_path')
        if poster_path:
            return f"https://image.tmdb.org/t/p/original/{poster_path}"
        else:
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching poster: {e}")
        return None


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

#Custom css for styling
st.markdown(
    """
    <style>
    .main-title {
        color: #007F5F;
        font-size: 45px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
    }
    .select-movie-label {
        color: #FF5733; 
        font-size: 24px;
    }
    .movie-caption {
        color: #1F2937; 
        font-size: 22px;
    }
    .recommended-movies-title {
        color: #2EC4B6; 
        font-size: 30px;
        margin-top: 5px;
        margin-bottom: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">🎬 Movie Recommender System</div>', unsafe_allow_html=True)

st.markdown("<div class='select-movie-label'>Select a movie</div>",unsafe_allow_html=True)
selected_movie_name = st.selectbox(
    'None',
    movies['title'].values,label_visibility="hidden"
)

if st.button("Show Recommendation"):
    with st.spinner("Fetching recommendations..."):
        recommendations, posters = recommend(selected_movie_name)
    
    if recommendations:
        st.markdown(f"<div class='recommended-movies-title'>Explore Movies Inspired by {selected_movie_name}</div>",unsafe_allow_html=True)
        cols = st.columns(2)
        for i,(movie, poster) in enumerate(zip(recommendations, posters)):
            with cols[i%2]:
                st.markdown(f"<div class='movie-caption'>{movie}</div>",unsafe_allow_html=True)
                st.image(poster,use_container_width=True)
    else:
        st.warning("No recommendations found!")