import sqlite3
import pandas as pd
from collections import defaultdict

df = pd.read_csv('steam_top_games_2026.csv')
connection = sqlite3.connect('sql_database.db')
cursor = connection.cursor()
df.to_sql("games", connection, if_exists="replace", index=False)

def normalize_genres(genre_string):
    return [
        g.strip().lower()
        for g in genre_string.split(',')]

user_input = input("make a top of ur games (use ',')")
games = [game.strip() for game in user_input.split(",")]
weighted_genres = []
for i,game in enumerate(games):
    cursor.execute("SELECT genres FROM games WHERE LOWER(name) LIKE LOWER(?)",('%' + game + '%',))
    result = cursor.fetchall()
    for r in result:
        genres = r[0]
        for g in genres.split(","):
            g = g.strip().lower()
            weight = 1 / (i + 1)
            weighted_genres.append((g, weight))

ser_gen = defaultdict(float)
for g, w in weighted_genres:
    ser_gen[g] += w

def game_scoring(row):
    score = 0
    if pd.isna(row["genres"]):
        return 0
    genres = row['genres'].split(',')
    for i in genres:
        i = i.strip().lower()
        if i in ser_gen:
            score += ser_gen[i]
    return score
df["score"] = df.apply(game_scoring, axis=1)
df_filtered = df[~df["name"].str.lower().isin([a.lower() for a in games])]
top_games = df_filtered.sort_values("score", ascending=False).head(10)
print(top_games[["name","score"]])