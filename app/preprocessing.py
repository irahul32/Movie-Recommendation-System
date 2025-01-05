import os
import pandas as pd
import numpy as np
import ast
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

#Initialize the stemmer
ps = PorterStemmer()

# get the absolute path of the project folder
BaseDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_data():
    """Load the dataset"""
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

def fetch_genkey(obj):
    """Extract 'name' values from a list of dictionaries(genres & keywords)."""
    L=[]
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

def fetch_casts(obj):
    """Extract the names of the top 5 cast members."""
    L=[]
    counter = 0
    for i in ast.literal_eval(obj):
        if counter!=5:
            L.append(i['name'])
            counter+=1
        else:
            break
    return L

def fetch_director(obj):
    """Extract the director's name from the crew column."""
    L=[]
    for i in ast.literal_eval(obj):
        if i['job'] == "Director":
            L.append(i['name'])
            break
    return L

def stem_text(text):
    """Apply stemming to a text"""
    y=[]
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

def prepare_data(movies_df):
    """Preprocess the movies dataset."""
    
    #getting the names of genres,keywords,casts and director in a list
    movies_df['genres']  = movies_df['genres'].apply(fetch_genkey)
    movies_df['keywords'] = movies_df['keywords'].apply(fetch_genkey)
    movies_df['cast'] =  movies_df['cast'].apply(fetch_casts)
    movies_df['crew'] = movies_df['crew'].apply(fetch_director)
    movies_df['overview'] = movies_df['overview'].apply(lambda x: x.split())

    #removing the sapaces between the names like Science Fiction to ScienceFiction
    movies_df['genres']  = movies_df['genres'].apply(lambda x : [i.replace(" ","") for i in x])
    movies_df['keywords'] = movies_df['keywords'].apply(lambda x : [i.replace(" ","") for i in x])
    movies_df['cast'] =  movies_df['cast'].apply(lambda x : [i.replace(" ","") for i in x])
    movies_df['crew'] = movies_df['crew'].apply(lambda x : [i.replace(" ","") for i in x])

    #Combine features into a single 'tags' column
    movies_df['tags'] = movies_df['overview'] + movies_df['genres'] + movies_df['keywords'] +       movies_df['cast'] + movies_df['crew']
    movies_df['tags'] = movies_df['tags'].apply(lamda x:" ".join(x).lower())

    #Apply Stemming
    movies_df['tags'] = movies_df['tags'].apply(stem_text)

    return movies_df[['movie_id','title','tags']]

def vectorize_data(new_df):
    """Vectorize the 'tags' column and compute similarity."""
    tf = TfidfVectorizer(max_features=5000,stop_words='english')
    vectors=tf.fit_transform(new_df['tags']).toarray()
    return movies,vectors

def preprocess_pipeline():
    """Full preprocessing pipeline"""
    movies_df = load_data()
    new_df = prepare_data(movies_df)
    return vectorize_data(new_df)