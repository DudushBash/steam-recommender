
A content-based recommendation system for Steam games using:
1.Genre-based filtering (v1)
2.TF-IDF vectorization on game tags (v2)

#Project Overview:

This project demonstrates how recommendation systems work using real Steam game data.
It evolves from a simple rule-based model to a machine learning approach using TF-IDF and cosine similarity.


#Versions:

## V1 - Genre-based recommender
- Uses genre frequency scoring
- Simple rule-based system
- Fast but less accurate

## V2 - TF-IDF + Cosine Similarity (ML approach)
- Converts game tags into vectors
- Builds user preference vector
- Uses cosine similarity for recommendations
- More accurate and scalable

#Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- SciPy

#How to run

bash
pip install -r requirements.txt
python v2_tfidf_tags/recommender.py
