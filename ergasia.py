import numpy as np
import scipy.io
import os
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import math
from sklearn import metrics
from scipy.stats import zscore
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn as sns
from datasets import load_dataset
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Set Kaggle API token
os.environ['KAGGLE_API_TOKEN'] = 'KGAT_d3d79d12cd6971f4edef5cb83dc2ba8d'

import kagglehub

# Download the dataset
path = kagglehub.dataset_download("abdallahwagih/imdb-movie-reviews")
print(f"Dataset downloaded to: {path}")

# Navigate to the aclImdb folder
acl_path = os.path.join(path, "aclImdb")

reviews = []
sentiments = []

# Load training reviews
for label, sentiment_value in [("pos", 1), ("neg", 0)]:
    folder_path = os.path.join(acl_path, "train", label)
    
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            if filename.endswith(".txt"):
                file_path = os.path.join(folder_path, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    reviews.append(f.read())
                    sentiments.append(sentiment_value)

# Create DataFrame
df = pd.DataFrame({'review': reviews, 'sentiment': sentiments})
print(f"Loaded {len(df)} reviews")
print(f"Positive: {sum(df['sentiment'])}")
print(f"Negative: {len(df) - sum(df['sentiment'])}")
print(df.head())

def setup_text_data(df, text_column='review', label_column='sentiment', max_features=1000):
    """
    Setup text data for clustering/analysis
    
    Parameters:
    - df: DataFrame with text reviews
    - text_column: name of column containing text reviews
    - label_column: name of column containing labels (for visualization)
    - max_features: number of features for TF-IDF (reduce for faster computation)
    """
    
    # Step 1: Extract text and labels
    X_text = df[text_column].values
    y = df[label_column].values if label_column in df.columns else None
    
    # Step 2: Convert text to numerical features using TF-IDF
    tfidf = TfidfVectorizer(max_features=max_features, stop_words='english')
    X_tfidf = tfidf.fit_transform(X_text).toarray()
    
    print(f"Original text data shape: {X_text.shape}")
    print(f"After TF-IDF vectorization: {X_tfidf.shape}")
    
    # Step 3: Standardize the TF-IDF features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_tfidf)
    
    # Step 4: Apply PCA for dimensionality reduction
    n_components = min(50, X_scaled.shape[1])
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    print(f"Explained variance ratio: {pca.explained_variance_ratio_[:10].sum():.2%}")
    
    # Step 5: Create DataFrame for visualization (use first 2 components)
    pca_df = pd.DataFrame(data=X_pca[:, :2], columns=['PC1', 'PC2'])
    if y is not None:
        pca_df['target'] = y
        pca_df['targetname'] = ['Positive' if i == 1 else 'Negative' for i in y]
    
    return {
        'X_scaled': X_scaled,
        'X_pca': X_pca,
        'pca_df': pca_df,
        'vectorizer': tfidf,
        'scaler': scaler,
        'pca': pca,
        'feature_names': tfidf.get_feature_names_out()
    }

# Setup the data for clustering
print("\n" + "="*50)
print("SETTING UP TEXT DATA FOR CLUSTERING")
print("="*50)

results = setup_text_data(df, text_column='review', label_column='sentiment', max_features=2000)

# Perform clustering on the PCA-reduced data
kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(results['X_pca'])

# Add cluster labels to the dataframe
results['pca_df']['cluster'] = clusters

# Visualize the clusters
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.scatterplot(data=results['pca_df'], x='PC1', y='PC2', hue='targetname', alpha=0.6)
plt.title('Actual Sentiment Labels')

plt.subplot(1, 2, 2)
sns.scatterplot(data=results['pca_df'], x='PC1', y='PC2', hue='cluster', alpha=0.6, palette='Set2')
plt.title('K-Means Clusters (k=2)')

plt.tight_layout()
plt.show()

# Calculate clustering performance
actual_labels = results['pca_df']['target']
ari = adjusted_rand_score(actual_labels, clusters)
nmi = normalized_mutual_info_score(actual_labels, clusters)

print(f"\nClustering Performance:")
print(f"Adjusted Rand Index: {ari:.3f}")
print(f"Normalized Mutual Info: {nmi:.3f}")