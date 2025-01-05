import os
import sys
from preprocessing import preprocess_pipeline
from sklearn.neighbors import NearestNeighbors

# Add the directory containing this script to the Python path
#CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
#sys.path.append(CURRENT_DIR)

#Load preprocessed data and vectors
new_df,vectors = preprocess_pipeline()

#Initialize and fit the NearestNeighbors model
model  = NearestNeighbors(metric='cosine')   #gives the cosine similarity
model.fit(vectors)

def recommend(movie):
    """
    Recommend movies similar to the given movie using Nearest Neighbors.
    Args:
        movie (str): Title of the movie for which recommendations are needed.
    Returns:
        List[str]: Titles of the recommended movies.
    """
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
    # Test the recommendation system
    test_movie = "Avatar"  # Replace with any movie title from the dataset
    recommendations = recommend(test_movie)
    print(f"Recommendations for '{test_movie}':")
    for movie in recommendations:
        print(movie)