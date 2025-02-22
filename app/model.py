import os
import sys
import numpy as np
from preprocessing import preprocess_pipeline
from sklearn.neighbors import NearestNeighbors
from joblib import load, dump

# Add the directory containing this script to the Python path
#CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
#sys.path.append(CURRENT_DIR)

def train_and_save_model(model_path, data_path, vector_path):
    """Trains the model and saves it to the specified path."""
    
    # Check if the model, data, and vectors already exist
    if os.path.exists(model_path) and os.path.exists(data_path) and os.path.exists(vector_path):
        print("Model and data already exist. Skipping training.")
        return

    # Create directories if they don't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    os.makedirs(os.path.dirname(vector_path), exist_ok=True)
    
    new_df, vectors = preprocess_pipeline()

    # Initialize and fit the Nearest Neighbors model
    model = NearestNeighbors(metric='cosine')
    model.fit(vectors)

    # Save the model
    dump(model, model_path)
    dump(new_df, data_path)
    np.save(vector_path, vectors)

def recommend(model_path, data_path, vector_path, movie):
    """
    Recommend movies similar to the given movie using Nearest Neighbors.
    Args:
        movie (str): Title of the movie for which recommendations are needed.
    Returns:
        List[str]: Titles of the recommended movies.
    """
    model = load(model_path)
    new_df = load(data_path)
    vectors = np.load(vector_path)

    try:
        movie_index = new_df[new_df['title'].str.lower() == movie.lower()].index[0]
    except IndexError:
        print(f"{movie} not found in the database. Please try another title.")
    
    #Get distances and indices of the nearest neighbors
    distances,indexes = model.kneighbors([vectors[movie_index]],n_neighbors=6)

    #Exclude the first recommendation (it is the same movie)
    recommended_movies = [new_df.iloc[i].title for i in indexes[0][1:]]

    return recommended_movies

# Example usage
if __name__ == "__main__":
    base_dir=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir,"model/knn_model.joblib")
    data_path = os.path.join(base_dir,"Dataset/preprocessed_data.joblib")
    vector_path = os.path.join(base_dir,"vector/vectors.npy")

    train_and_save_model(model_path, data_path, vector_path)
    print("Model trained and saved.")

    # Test the recommendation system
    test_movie = "Avatar"  # Replace with any movie title from the dataset
    recommendations = recommend(model_path, data_path, vector_path, test_movie)
    print(f"Recommendations for '{test_movie}':")
    for movie in recommendations:
        print(movie)