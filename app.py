import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ✅ SET PAGE CONFIG AT THE VERY TOP
st.set_page_config(page_title="🎬 Movie Recommender", layout="centered")

# -------------------------- Data Loading --------------------------
@st.cache_data
def load_data():
    movies = pd.read_csv("movies.csv", encoding='latin-1')
    ratings = pd.read_csv("ratings.csv", encoding='latin-1')
    tags = pd.read_csv("tags.csv", encoding='latin-1')
    return movies, ratings, tags

# -------------------------- Preprocessing --------------------------
def preprocess_data(movies, ratings):
    ratings = ratings.merge(movies, on='movieId', how='left')
    
    # Remove outliers
    Q1 = ratings['rating'].quantile(0.25)
    Q3 = ratings['rating'].quantile(0.75)
    IQR = Q3 - Q1
    ratings = ratings[(ratings['rating'] >= Q1 - 1.5*IQR) & (ratings['rating'] <= Q3 + 1.5*IQR)]

    # User-Movie Matrix
    user_movie_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
    scaler = MinMaxScaler()
    user_movie_matrix_scaled = scaler.fit_transform(user_movie_matrix)

    return ratings, user_movie_matrix, user_movie_matrix_scaled

ratings, user_movie_matrix, user_movie_matrix_scaled = preprocess_data(movies, ratings)

# -------------------------- Collaborative Filtering (SVD) --------------------------
svd = TruncatedSVD(n_components=10)
svd_matrix = svd.fit_transform(user_movie_matrix_scaled)
reconstructed = np.dot(svd_matrix, svd.components_)
rmse_svd = np.sqrt(mean_squared_error(user_movie_matrix_scaled, reconstructed))

# -------------------------- Content-Based Filtering --------------------------
movies['genres'] = movies['genres'].fillna('')
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(movies['genres'])
genre_sim = cosine_similarity(tfidf_matrix)

def content_based_recommend(title, cosine_sim, movies_df):
    indices = pd.Series(movies_df.index, index=movies_df['title']).drop_duplicates()
    if title not in indices:
        return ["Movie not found."]
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    movie_indices = [i[0] for i in sim_scores]
    return movies_df['title'].iloc[movie_indices].tolist()

# -------------------------- Hybrid Filtering --------------------------
# NOTE: Shape mismatch is possible; we handle it by trimming
min_rows = min(reconstructed.shape[0], genre_sim.shape[0])
min_cols = min(reconstructed.shape[1], genre_sim.shape[1])
hybrid_matrix = 0.6 * reconstructed[:min_rows, :min_cols] + 0.4 * genre_sim[:min_rows, :min_cols]
rmse_hybrid = np.sqrt(mean_squared_error(user_movie_matrix_scaled[:min_rows, :min_cols], hybrid_matrix))

# -------------------------- Classification --------------------------
movie_data = ratings.copy()
movie_data['label'] = (movie_data['rating'] >= 3.5).astype(int)
X = svd_matrix
y = movie_data['label'].values[:X.shape[0]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')

# -------------------------- Streamlit UI --------------------------
st.set_page_config(page_title="🎬 Movie Recommender", layout="centered")
st.title("🎥 Movie Recommendation System")

menu = ["Content-Based", "Collaborative Filtering", "Hybrid", "Model Evaluation"]
choice = st.sidebar.selectbox("Choose Recommendation Type", menu)

if choice == "Content-Based":
    st.subheader("🔍 Content-Based Recommendation")
    movie_list = movies['title'].dropna().unique().tolist()
    selected_movie = st.selectbox("Choose a movie", movie_list)

    if st.button("Recommend"):
        recommendations = content_based_recommend(selected_movie, genre_sim, movies)
        st.write("Top 5 Recommended Movies:")
        st.table(recommendations)

elif choice == "Collaborative Filtering":
    st.subheader("👥 Collaborative Filtering (SVD)")
    st.write(f"RMSE from SVD Model: {rmse_svd:.4f}")
    st.info("Collaborative predictions can be added here with user input.")

elif choice == "Hybrid":
    st.subheader("🔗 Hybrid Recommendation")
    st.write(f"RMSE from Hybrid Model: {rmse_hybrid:.4f}")
    st.info("This combines collaborative and content-based methods. Personalization can be added.")

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

