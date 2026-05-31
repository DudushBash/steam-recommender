import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
import joblib
import os

df = pd.read_csv('steam_top_games_2026.csv')
df["tags"] = df["tags"].fillna('')
vectorizer = TfidfVectorizer()
if os.path.exists("tfidf_matrix.npz"):
    vectorizer = joblib.load("vectorizer.pkl")
    tfidf_matrix = sparse.load_npz("tfidf_matrix.npz")
else:
    tfidf_matrix = vectorizer.fit_transform(df["tags"])
    joblib.dump(vectorizer, "vectorizer.pkl")
    sparse.save_npz("tfidf_matrix.npz", tfidf_matrix)
    
userin = input("input games (use ',') ")
games = []
for g in userin.split(','):
    games.append(g.strip())
us_indx = df[df["name"].str.lower().isin([g.lower() for g in games])].index
us_vectors =  tfidf_matrix[us_indx]
pocti_usvec = us_vectors.mean(axis=0)
us_vector = np.asarray(pocti_usvec)
poxojest = cosine_similarity(us_vector, tfidf_matrix)[0]
df['similarity'] = poxojest
df_filt = df[~df["name"].str.lower().isin([g.lower() for g in games])]
top = df_filt.sort_values('similarity', ascending=False).head(10)
print(top[["name", "similarity"]])