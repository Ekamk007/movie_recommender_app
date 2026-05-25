import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
import gdown

# ── Page Config ─────────────────────────────────────────────
st.set_page_config(
    page_title="CineMatch",
    page_icon="🎬",
    layout="wide"
)

# ── Custom CSS ──────────────────────────────────────────────
st.markdown("""
<style>

.main {
    background-color: #0f1117;
}

.hero {
    text-align: center;
    padding: 20px;
    margin-bottom: 20px;
}

.hero h1 {
    color: white;
    font-size: 3rem;
}

.hero p {
    color: #9ca3af;
}

.movie-card {
    background: #1a1a2e;
    border-radius: 14px;
    padding: 12px;
    text-align: center;
    transition: 0.3s;
    height: 100%;
    border: 1px solid #2d2d44;
}

.movie-card:hover {
    transform: translateY(-5px);
    box-shadow: 0px 10px 25px rgba(0,0,0,0.4);
}

.movie-card img {
    width: 100%;
    border-radius: 10px;
}

.title {
    color: white;
    font-weight: 700;
    margin-top: 10px;
    font-size: 1rem;
}
.metric {
    color: #d1d5db;
    margin-top: 6px;
    font-size: 0.9rem;
}

.score {
    color: #60a5fa;
    font-weight: bold;
    margin-top: 8px;
    font-size: 1rem;
}

.review-box {
    background: #111827;
    padding: 10px;
    border-radius: 10px;
    color: #d1d5db;
    font-size: 0.85rem;
    margin-top: 10px;
    text-align: left;
}

</style>
""", unsafe_allow_html=True)

# ── Download similarity.pkl via gdown ────────────────────────
FILE_ID = "1wdGcsHwkjJEt-81LuixSXep8Li3Jww3p"
OUTPUT = "similarity.pkl"

if not os.path.exists(OUTPUT):
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    gdown.download(url, OUTPUT, quiet=False)

# ── Constants ───────────────────────────────────────────────
FALLBACK_IMG = "https://via.placeholder.com/300x450?text=No+Poster"

# ── Load Data ───────────────────────────────────────────────
@st.cache_resource
def load_data():
    movies = pickle.load(open("movies_enriched.pkl", "rb"))
    similarity = pickle.load(open("similarity.pkl", "rb"))
    return movies, similarity

movies_df, similarity = load_data()

# ── Sentiment Analysis ──────────────────────────────────────
def simple_sentiment(reviews):
    if not reviews:
        return 0.5

    pos = {
        "good", "great", "amazing", "love", "loved", "best",
        "awesome", "excellent", "fantastic", "masterpiece",
        "brilliant", "wonderful", "enjoy", "enjoyed", "perfect",
        "beautiful", "compelling", "outstanding", "superb", "fun"
    }

    neg = {
        "bad", "worst", "boring", "hate", "hated",
        "awful", "waste", "poor", "terrible", "disappointing",
        "dull", "weak", "mediocre", "slow", "overrated",
        "predictable", "stupid", "annoying", "forgettable", "bland"
    }

    scores = []

    for r in reviews[:5]:
        words = r.lower().split()
        # Strip punctuation from words
        words = [w.strip(".,!?\"'") for w in words]

        p = sum(w in pos for w in words)
        n = sum(w in neg for w in words)

        scores.append(p / (p + n) if (p + n) > 0 else 0.5)

    return round(sum(scores) / len(scores), 3)

# ── Recommendation Engine ───────────────────────────────────
def recommend(movie_title, top_n=5):

    idx = movies_df[movies_df["title"] == movie_title].index[0]

    distances = similarity[idx]

    candidates = sorted(
        enumerate(distances),
        key=lambda x: x[1],
        reverse=True
    )[1:20]

    results = []

    for i, sim in candidates:

        row = movies_df.iloc[i]

        reviews = row.get("reviews", [])

        sentiment = simple_sentiment(reviews)

        final_score = (0.8 * sim) + (0.2 * sentiment)

        results.append({
            "title": row["title"],
            "poster": row.get("poster", FALLBACK_IMG),
            "reviews": reviews,
            "similarity": round(float(sim), 3),
            "sentiment": round(sentiment, 3),
            "final": round(final_score, 3)
        })

    results.sort(key=lambda x: x["final"], reverse=True)

    return results[:top_n]

# ── Hero Section ────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🎬 CineMatch</h1>
    <p>AI Powered Offline Movie Recommendation System</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ─────────────────────────────────────────────────
with st.sidebar:

    st.header("⚙️ Settings")

    top_n = st.slider(
        "Number of Recommendations",
        min_value=3,
        max_value=10,
        value=5
    )

    show_reviews = st.toggle("Show Reviews", value=True)

    st.markdown("---")

    st.markdown("""
    ### About
    - Content Based Filtering
    - Cosine Similarity
    - Sentiment Aware Ranking
    - Fully Offline System
    """)

# ── Movie Selector ──────────────────────────────────────────
movie_list = sorted(movies_df["title"].values)

selected = st.selectbox(
    "🔍 Search Movie",
    movie_list
)

# ── Selected Movie Display ──────────────────────────────────
selected_row = movies_df[movies_df["title"] == selected].iloc[0]

st.subheader("Selected Movie")

col1, col2 = st.columns([1, 3])

with col1:
    st.image(selected_row.get("poster", FALLBACK_IMG), width=220)

with col2:
    st.markdown(f"## {selected}")

# ── Recommendation Button ───────────────────────────────────
if st.button("🎯 Recommend Movies"):

    with st.spinner("Generating recommendations..."):

        results = recommend(selected, top_n)

    st.markdown("## 🍿 Recommended Movies")

    cols = st.columns(min(top_n, 5))

    for idx, movie in enumerate(results):

        with cols[idx % 5]:                          # BUG 2 & 3 FIX: all card content inside column block

            mood = int(movie["sentiment"] * 100)

            card_html = f"""
<div class="movie-card">
    <img src="{movie['poster']}">
    <div class="title">{movie['title']}</div>
    <div class="metric">Similarity: {movie['similarity']}</div>
    <div class="metric">Mood Score: {mood}%</div>
    <div class="score">Final Score: {movie['final']}</div>
</div>
"""                                                  # BUG 1 FIX: removed leading space before card_html

            st.markdown(card_html, unsafe_allow_html=True)

            st.progress(movie["sentiment"])

            if show_reviews and movie["reviews"]:

                with st.expander("Reviews Preview"):

                    for r in movie["reviews"][:2]:

                        st.markdown(f"""
                        <div class="review-box">
                        {r[:300]}...
                        </div>
                        """, unsafe_allow_html=True)

    # ── Analytics Table ─────────────────────────────────────
    st.markdown("---")

    st.subheader("📊 Recommendation Analytics")

    analytics_df = pd.DataFrame([{
        "Movie": m["title"],
        "Similarity": m["similarity"],
        "Sentiment": m["sentiment"],
        "Final Score": m["final"]
    } for m in results])

    st.dataframe(
        analytics_df,
        use_container_width=True,
        hide_index=True
    )

    csv = analytics_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇ Download Recommendations CSV",
        csv,
        "recommendations.csv",
        "text/csv"
    )

# ── Footer ──────────────────────────────────────────────────
st.markdown("---")

st.caption(
    "CineMatch • Offline Movie Recommendation System • Streamlit + NLP + Cosine Similarity"
)
