import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ---------------------- Page Config ----------------------
st.set_page_config(page_title="🎬 Movie Recommender", layout="centered")

# ---------------------- Load Data ----------------------
@st.cache_data
def load_data():
    movies = pd.read_csv("movies.csv", encoding='latin-1')
    ratings = pd.read_csv("ratings.csv", encoding='latin-1')
    tags = pd.read_csv("tags.csv", encoding='latin-1', on_bad_lines='skip')
    return movies, ratings, tags

# ---------------------- Preprocessing ----------------------
def preprocess_data(movies, ratings):
    merged = ratings.merge(movies, on='movieId', how='left')
    
    # Remove rating outliers
    Q1 = merged['rating'].quantile(0.25)
    Q3 = merged['rating'].quantile(0.75)
    IQR = Q3 - Q1
    filtered = merged[(merged['rating'] >= Q1 - 1.5*IQR) & (merged['rating'] <= Q3 + 1.5*IQR)]

    # Create user-movie matrix
    matrix = filtered.pivot(index='userId', columns='movieId', values='rating').fillna(0)
    scaled_matrix = MinMaxScaler().fit_transform(matrix)

    return filtered, matrix, scaled_matrix

# ---------------------- SVD ----------------------
def perform_svd(scaled_matrix):
    svd = TruncatedSVD(n_components=10)
    svd_matrix = svd.fit_transform(scaled_matrix)
    reconstructed = np.dot(svd_matrix, svd.components_)
    rmse = np.sqrt(mean_squared_error(scaled_matrix, reconstructed))
    return svd_matrix, reconstructed, rmse

# ---------------------- Content-Based ----------------------
def content_based_recommend(title, movies, cosine_sim):
    indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()
    if title not in indices:
        return ["Movie not found."]
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices].tolist()

# ---------------------- Hybrid Filtering ----------------------
def hybrid_filter(reconstructed, genre_sim, original_matrix):
    min_rows = min(reconstructed.shape[0], genre_sim.shape[0])
    min_cols = min(reconstructed.shape[1], genre_sim.shape[1])
    hybrid = 0.6 * reconstructed[:min_rows, :min_cols] + 0.4 * genre_sim[:min_rows, :min_cols]
    rmse = np.sqrt(mean_squared_error(original_matrix[:min_rows, :min_cols], hybrid))
    return hybrid, rmse

# ---------------------- Classification ----------------------
def classify_users(svd_matrix, filtered_ratings):
    filtered_ratings['label'] = (filtered_ratings['rating'] >= 3.5).astype(int)
    X = svd_matrix
    y = filtered_ratings['label'].values[:X.shape[0]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')

    return y_test, y_pred, cv_scores

# ---------------------- UI ----------------------
st.title("🎥 Movie Recommendation System")

menu = ["Content-Based", "Collaborative Filtering", "Hybrid", "Model Evaluation"]
choice = st.sidebar.selectbox("Choose Recommendation Type", menu)

# Load and preprocess
movies, ratings, tags = load_data()
filtered_ratings, matrix, scaled_matrix = preprocess_data(movies, ratings)
svd_matrix, reconstructed, svd_rmse = perform_svd(scaled_matrix)

# Content-Based
movies['genres'] = movies['genres'].fillna('')
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(movies['genres'])
cosine_sim = cosine_similarity(tfidf_matrix)

# Hybrid
hybrid_matrix, hybrid_rmse = hybrid_filter(reconstructed, cosine_sim, scaled_matrix)

# Classification
y_test, y_pred, cv_scores = classify_users(svd_matrix, filtered_ratings)

# ---------------------- Streamlit Sections ----------------------
if choice == "Content-Based":
    st.subheader("🔍 Content-Based Recommendation")
    movie_titles = movies['title'].dropna().unique()
    selected_movie = st.selectbox("Choose a movie", sorted(movie_titles))

    if st.button("Recommend"):
        recommendations = content_based_recommend(selected_movie, movies, cosine_sim)
        st.write("Top 5 similar movies:")
        st.table(recommendations)

elif choice == "Collaborative Filtering":
    st.subheader("👥 Collaborative Filtering (SVD)")
    st.write(f"RMSE of SVD Model: `{svd_rmse:.4f}`")

elif choice == "Hybrid":
    st.subheader("🔗 Hybrid Recommendation")
    st.write(f"RMSE of Hybrid Model: `{hybrid_rmse:.4f}`")

elif choice == "Model Evaluation":
    st.subheader("📊 Logistic Regression Evaluation")
    st.write("Accuracy:", accuracy_score(y_test, y_pred))
    st.write("Precision:", precision_score(y_test, y_pred))
    st.write("Recall:", recall_score(y_test, y_pred))
    st.write("F1 Score:", f1_score(y_test, y_pred))
    st.write("Confusion Matrix:")
    st.write(confusion_matrix(y_test, y_pred))
    st.write("Cross-Validation Accuracy Scores:", cv_scores)
    st.write("Mean CV Accuracy:", np.mean(cv_scores))

st.markdown("---")
st.caption("Built with ❤️ using Streamlit")
