import streamlit as st
import numpy as np
import pandas as pd
import pickle
from pathlib import Path

# ── Page Config ─────────────────────────────
st.set_page_config(
    page_title="CineMatch",
    page_icon="🎬",
    layout="wide"
)

# ── UI ──────────────────────────────────────
st.markdown("""
<style>
.movie-card {
    background: #1a1a2e;
    border-radius: 12px;
    padding: 10px;
    text-align: center;
}
.movie-card img {
    width: 100%;
    border-radius: 10px;
}
.title {
    color: white;
    font-weight: 600;
    margin-top: 5px;
}
</style>
""", unsafe_allow_html=True)

FALLBACK_IMG = "https://via.placeholder.com/300x450?text=No+Poster"

# ── Load Data ───────────────────────────────
@st.cache_resource
def load_data():
    movies = pickle.load(open("movies_enriched.pkl", "rb"))
    similarity = pickle.load(open("similarity.pkl", "rb"))
    return movies, similarity

movies_df, similarity = load_data()

# ── Sentiment ───────────────────────────────
def simple_sentiment(reviews):
    if not reviews:
        return 0.5

    pos = {"good","great","amazing","love","best","awesome","excellent"}
    neg = {"bad","worst","boring","hate","awful","waste","poor"}

    scores = []
    for r in reviews[:5]:
        words = r.lower().split()
        p = sum(w in pos for w in words)
        n = sum(w in neg for w in words)
        scores.append(p/(p+n) if (p+n)>0 else 0.5)

    return sum(scores)/len(scores)

# ── Recommendation ──────────────────────────
def recommend(movie_title, top_n=5):

    idx = movies_df[movies_df["title"] == movie_title].index[0]
    distances = similarity[idx]

    candidates = sorted(
        enumerate(distances),
        key=lambda x: x[1],
        reverse=True
    )[1:15]

    results = []

    for i, sim in candidates:
        row = movies_df.iloc[i]

        sentiment = simple_sentiment(row["reviews"])
        final_score = 0.8 * sim + 0.2 * sentiment

        results.append({
            "title": row["title"],
            "poster": row.get("poster", FALLBACK_IMG),
            "similarity": round(sim, 3),
            "sentiment": round(sentiment, 3),
            "final": round(final_score, 3)
        })

    results.sort(key=lambda x: x["final"], reverse=True)
    return results[:top_n]

# ── App Title ───────────────────────────────
st.title("🎬 CineMatch - Movie Recommender")

movie_list = sorted(movies_df["title"].values)
selected = st.selectbox("Select a movie", movie_list)

# ── Button ──────────────────────────────────
if st.button("Recommend"):

    results = recommend(selected)

    cols = st.columns(len(results))

    for i, movie in enumerate(results):
        with cols[i]:

            st.markdown(f"""
            <div class="movie-card">
                <img src="{movie['poster']}" />
                <div class="title">{movie['title']}</div>
                <small>Sim: {movie['similarity']}</small><br>
                <small>Mood: {movie['sentiment']}</small><br>
                <b>Score: {movie['final']}</b>
            </div>
            """, unsafe_allow_html=True)