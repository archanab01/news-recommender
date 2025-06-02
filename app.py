# news_recommender_app.py

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load and preprocess data
@st.cache_data

def load_data():
    df = pd.read_csv("articles.csv")
    df = df[['Article_Id', 'Title', 'Author', 'Content', 'URL']]
    df.dropna(subset=['Title', 'Content'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def vectorize_content(df):
    tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)
    tfidf_matrix = tfidf.fit_transform(df['Content'])
    return tfidf_matrix

def recommend_articles(title, df, tfidf_matrix, top_n=5):
    title = title.strip().lower()
    matches = df[df['Title'].str.lower().str.contains(title)]
    if matches.empty:
        return []
    idx = matches.index[0]  # Take first matching article
    cosine_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    similar_indices = cosine_sim.argsort()[-top_n-1:-1][::-1]  # Exclude input article
    return df.iloc[similar_indices][['Title', 'Author', 'URL']]

# Streamlit UI
st.title("📰 News Article Recommender")

st.markdown("""
Enter a part of the **news article title** to find similar articles.
""")

# Load data
news_df = load_data()
tfidf_matrix = vectorize_content(news_df)

# User input
user_input = st.text_input("Enter news article title")

if user_input:
    results = recommend_articles(user_input, news_df, tfidf_matrix)
    if len(results):
        st.subheader("🔍 Recommended Articles:")
        for i, row in results.iterrows():
            st.markdown(f"**{row['Title']}**")
            st.markdown(f"*Author:* {row['Author']}")
            if pd.notna(row['URL']):
                st.markdown(f"[Read more]({row['URL']})")
            st.markdown("---")
    else:
        st.warning("No matching articles found. Try a different keyword.")
