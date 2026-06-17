import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "..", "DATA", "steam_top_games_2026.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
matrix_path = os.path.join(MODEL_DIR, "tfidf_matrix.npz")
vectorizer_path = os.path.join(MODEL_DIR, "vectorizer.pkl")

df = pd.read_csv(csv_path)
df["tags"] = df["tags"].fillna('')
df["review_score"] = (df["positive_reviews"] /(df["positive_reviews"] + df["negative_reviews"])).fillna(df["review_score"].mean())  # процент "качества"
vectorizer = TfidfVectorizer()
if os.path.exists(matrix_path):
    vectorizer = joblib.load(vectorizer_path)
    tfidf_matrix = sparse.load_npz(matrix_path)
else:
    tfidf_matrix = vectorizer.fit_transform(df["tags"])
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(vectorizer, vectorizer_path)
    sparse.save_npz(matrix_path, tfidf_matrix)
    
userin = input("input games (use ',') ")
games = []
for g in userin.split(','):
    games.append(g.strip())
us_indx = df[df["name"].str.lower().isin([g.lower() for g in games])].index
us_vectors =  tfidf_matrix[us_indx]
us_vector = np.asarray(us_vectors.mean(axis=0))
poxojest = cosine_similarity(us_vector, tfidf_matrix)[0]
df['similarity'] = poxojest
df['final_score'] = (0.7 * df['similarity'] + 0.3 * df['review_score'])
df_filt = df[~df["name"].str.lower().isin([g.lower() for g in games])]
top = df_filt.sort_values('final_score', ascending=False).head(10)
print(top[["name", "final_score"]])
