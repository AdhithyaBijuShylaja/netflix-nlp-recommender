import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

st.set_page_config(page_title="Netflix Recommender", layout="centered")

st.title("🎬 Netflix Show Recommender")
st.write("Get similar show recommendations using description-based NLP")

@st.cache_data
def load_data():
    df = pd.read_csv("netflix_titles.csv")
    df = df[['title', 'description']].dropna().drop_duplicates().reset_index()
    return df

df = load_data()

@st.cache_resource
def get_similarity_matrix(data):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(data['description'])
    return linear_kernel(tfidf_matrix, tfidf_matrix)

cosine_sim = get_similarity_matrix(df)

def recommend(title):
    title = title.lower()
    if title not in df['title'].str.lower().values:
        return ["❌ Show not found in database."]
    idx = df[df['title'].str.lower() == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    show_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[show_indices].tolist()

user_input = st.text_input("Enter a show name:", "")

if user_input:
    st.subheader("📢 Recommended Shows:")
    for show in recommend(user_input):
        st.write(f"✅ {show}")
