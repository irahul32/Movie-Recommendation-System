import streamlit as st
from joblib import load, dump

model = load("model/knn_model.joblib")
movies_df = load("Dataset/preprocessed_data.joblib")

st.title("Movie Recommender System")
st.text("Select a movie")