import os
import pandas as pd
import numpy as np
import ast

# get the absolute path of the project folder
BaseDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_data():
    try:
        #construct file paths
        movies_path = os.path.join(BaseDir,"Dataset","tmdb_5000_movies.csv")
        credits_path = os.path.join(BaseDir,"Dataset","tmdb_5000_credits.csv")

        # Check if files exist
        if not os.path.exists(movies_path):
            raise FileNotFoundError(f"File not found: {movies_path}")
        if not os.path.exists(credits_path):
            raise FileNotFoundError(f"File not found: {credits_path}")
        
        #load datasets
        movies = pd.read_csv(movies_path)
        credits = pd.read_csv(credits_path)
    except pd.errors.EmptyDataError as e:
        raise ValueError(
            "One or both dataset files are empty. Please check the files"
        ) from e
    except Exception as e:
        raise Exception(f"An error occurred while loading datasets: {e}") from e
    
    #Merged datasets
    movies = movies.merge(credits,on='title')

    #keep necessary columns
    movies_df = movies[['movie_id','title','overview','genres','keywords','cast','crew']]
    
    #Handling missing values
    movies_df.dropna(inplace=True)

    return movies_df

#fetching genres and keywords name
def fetch_genkey(obj):
    L=[]
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

#fetching top 5 casts name
def fetch_casts(obj):
    L=[]
    counter = 0
    for i in ast.literal_eval(obj):
        if counter!=5:
            L.append(i['name'])
            counter+=1
        else:
            break
    return L

#fetching director's name 
def fetch_director(obj):
    L=[]
    for i in ast.literal_eval(obj):
        if i['job'] == "Director":
            L.append(i['name'])
            break
    return L
        


    