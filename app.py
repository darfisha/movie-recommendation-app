
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold

# Load data
movies = pd.read_csv("movies.csv", encoding='latin-1')
ratings = pd.read_csv("ratings.csv", encoding='latin-1')
tags = pd.read_csv("tags.csv", encoding='latin-1')

# Merge
ratings = ratings.merge(movies, on='movieId', how='left')
tags = tags.merge(movies, on='movieId', how='left')

# Remove outliers in rating
Q1 = ratings['rating'].quantile(0.25)
Q3 = ratings['rating'].quantile(0.75)
IQR = Q3 - Q1
ratings = ratings[(ratings['rating'] >= Q1 - 1.5*IQR) & (ratings['rating'] <= Q3 + 1.5*IQR)]

# Create user-movie matrix
user_movie_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
scaler = MinMaxScaler()
user_movie_matrix_scaled = scaler.fit_transform(user_movie_matrix)

# SVD
svd = TruncatedSVD(n_components=10)
svd_matrix = svd.fit_transform(user_movie_matrix_scaled)
reconstructed = np.dot(svd_matrix, svd.components_)
rmse_svd = np.sqrt(mean_squared_error(user_movie_matrix_scaled, reconstructed))

# TF-IDF (Content-based)
vectorizer = TfidfVectorizer(stop_words='english')
movies['genres'] = movies['genres'].fillna('')
tfidf_matrix = vectorizer.fit_transform(movies['genres'])
genre_sim = cosine_similarity(tfidf_matrix)

# Hybrid
final_prediction = 0.6 * reconstructed + 0.4 * genre_sim[:reconstructed.shape[0], :reconstructed.shape[1]]
rmse_hybrid = np.sqrt(mean_squared_error(user_movie_matrix_scaled, final_prediction))

# Classification
movie_data = ratings.copy()
movie_data['label'] = (movie_data['rating'] >= 3.5).astype(int)
X = svd_matrix
y = movie_data['label'].values[:X.shape[0]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Streamlit UI
st.set_page_config(page_title="🎬 Movie Recommender", layout="centered")
st.title("🎥 Movie Recommendation System")

menu = ["Content-Based", "Collaborative Filtering", "Hybrid", "Model Evaluation"]
choice = st.sidebar.selectbox("Choose Recommendation Type", menu)

def content_based_recommend(movie_title, cosine_sim, indices):
    idx = indices[movie_title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices]

if choice == "Content-Based":
    st.subheader("🔍 Content-Based Recommendation")
    movie_list = movies['title'].dropna().unique().tolist()
    selected_movie = st.selectbox("Choose a movie", movie_list)

    if st.button("Recommend"):
        indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()
        recommendations = content_based_recommend(selected_movie, genre_sim, indices)
        st.write("Top 5 Recommended Movies:")
        st.table(recommendations)

elif choice == "Collaborative Filtering":
    st.subheader("👥 Collaborative Filtering")
    st.write(f"RMSE from SVD: {rmse_svd:.4f}")
    st.info("Collaborative recommendations can be displayed here with user input.")

elif choice == "Hybrid":
    st.subheader("🔗 Hybrid Recommendation")
    st.write(f"RMSE from Hybrid Model: {rmse_hybrid:.4f}")
    st.info("You can blend scores and display personalized results here.")

elif choice == "Model Evaluation":
    st.subheader("📊 Logistic Regression Evaluation")
    st.write("Accuracy:", accuracy_score(y_test, y_pred))
    st.write("Precision:", precision_score(y_test, y_pred))
    st.write("Recall:", recall_score(y_test, y_pred))
    st.write("F1 Score:", f1_score(y_test, y_pred))
    st.write("Confusion Matrix:")
    st.write(confusion_matrix(y_test, y_pred))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
    st.write("Cross-Validation Accuracy Scores:", cv_scores)
    st.write("Mean CV Accuracy:", np.mean(cv_scores))

st.markdown("---")
st.caption("Built with ❤️ using Streamlit")
