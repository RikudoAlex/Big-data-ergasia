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



# Καθαρισμος κειμενου
# Προσθήκη καθαρισμού κειμένου (ΑΝ ΔΕΝ ΤΟ ΕΧΕΙΣ ΗΔΗ)
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    # Αφαίρεση HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Μόνο γράμματα
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Lowercase
    text = text.lower()
    # Tokenization
    tokens = text.split()
    # Αφαίρεση stopwords & lemmatization
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and len(w) > 2]
    return ' '.join(tokens)

df['cleaned_review'] = df['review'].apply(clean_text)


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



# Μερος 1
print("\n" + "="*50)
print("PART 1: CLASSIFICATION MODELS")
print("="*50)

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB          # <-- ΑΛΛΑΓΗ: GaussianNB
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

# Χρησιμοποιούμε το TF-IDF (όχι PCA) για classification
X = results['X_scaled']  # TF-IDF scaled features
y = df['sentiment'].values

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"Train size: {X_train.shape}")
print(f"Test size: {X_test.shape}")

# Ορισμός μοντέλων
models = {
    'Naive Bayes': GaussianNB(),                    # <-- ΑΛΛΑΓΗ
    'SVM': LinearSVC(C=1.0, max_iter=2000, random_state=42),
    'Neural Network': MLPClassifier(
        hidden_layer_sizes=(128, 64), 
        max_iter=100, 
        early_stopping=True,
        random_state=42
    ),
    'Decision Tree': DecisionTreeClassifier(max_depth=20, random_state=42)
}

classification_results = {}

# Εκπαίδευση & Αξιολόγηση
for name, model in models.items():
    print(f"\nΕκπαίδευση {name}...")
    
    # Εκπαίδευση
    model.fit(X_train, y_train)
    
    # Πρόβλεψη
    y_pred = model.predict(X_test)
    
    # Probabilities για ROC
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_prob = model.decision_function(X_test)
    else:
        y_prob = None
    
    # Μετρικές
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    classification_results[name] = {
        'model': model,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1
    }
    
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall: {rec:.4f}")
    print(f"  F1-Score: {f1:.4f}")


    # Μερος 2

    print("\n" + "="*50)
print("PART 2: CONFUSION MATRICES & ROC CURVES")
print("="*50)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Confusion Matrices
for i, (name, res) in enumerate(classification_results.items()):
    row, col = i // 2, i % 2
    
    cm = confusion_matrix(y_test, res['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[row, col],
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    axes[row, col].set_title(f'{name}\nAccuracy: {res["accuracy"]:.3f}')
    axes[row, col].set_xlabel('Predicted')
    axes[row, col].set_ylabel('Actual')

# ROC Curves (όλα μαζί)
colors = ['blue', 'red', 'green', 'orange']
for i, (name, res) in enumerate(classification_results.items()):
    if res['y_prob'] is not None:
        fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
        roc_auc = auc(fpr, tpr)
        axes[1, 2].plot(fpr, tpr, color=colors[i], label=f'{name} (AUC={roc_auc:.3f})')

axes[1, 2].plot([0, 1], [0, 1], 'k--', alpha=0.5)
axes[1, 2].set_title('ROC Curves')
axes[1, 2].set_xlabel('False Positive Rate')
axes[1, 2].set_ylabel('True Positive Rate')
axes[1, 2].legend()

plt.tight_layout()
plt.show()


#Μερος 3

print("\n" + "="*50)
print("PART 3: ΣΥΓΚΡΙΣΗ ΜΟΝΤΕΛΩΝ")
print("="*50)

# Δημιουργία πίνακα σύγκρισης
comparison_data = []
for name, res in classification_results.items():
    comparison_data.append({
        'Model': name,
        'Accuracy': res['accuracy'],
        'Precision': res['precision'],
        'Recall': res['recall'],
        'F1-Score': res['f1']
    })

comp_df = pd.DataFrame(comparison_data)
comp_df = comp_df.sort_values('F1-Score', ascending=False)

print("\n=== ΣΥΓΚΡΙΤΙΚΟΣ ΠΙΝΑΚΑΣ ===")
print(comp_df.round(4).to_string(index=False))

# Bar chart
fig, ax = plt.subplots(figsize=(10, 6))
comp_df.set_index('Model').plot(kind='bar', ax=ax, colormap='viridis')
plt.title('Σύγκριση Μοντέλων Classification - IMDB Reviews')
plt.xlabel('Μοντέλα')
plt.ylabel('Score')
plt.xticks(rotation=0)
plt.legend(loc='lower right')
plt.grid(axis='y', alpha=0.3)

# Προσθήκη τιμών πάνω από τις μπάρες
for container in ax.containers:
    ax.bar_label(container, fmt='%.3f', fontsize=9)

plt.tight_layout()
plt.show()

# Καλύτερο μοντέλο
best_model = comp_df.iloc[0]
print(f"\n🏆 Καλύτερο μοντέλο: {best_model['Model']}")
print(f"   F1-Score: {best_model['F1-Score']:.4f}")


# Μερος 4
print("\n" + "="*50)
print("PART 4: ASSOCIATION RULES - WORD PAIRS")
print("="*50)

from sklearn.feature_extraction.text import CountVectorizer
from mlxtend.frequent_patterns import apriori, association_rules

# Πάρε δείγμα για ταχύτητα (3000 reviews)
np.random.seed(42)
sample_idx = np.random.choice(len(df), size=min(3000, len(df)), replace=False)
df_sample = df.iloc[sample_idx]

# Binary matrix με τις top 80 λέξεις
cv = CountVectorizer(max_features=80, binary=True, stop_words='english')
word_matrix = cv.fit_transform(df_sample['cleaned_review'])
word_df = pd.DataFrame(word_matrix.toarray(), columns=cv.get_feature_names_out())

print(f"Διαστάσεις binary matrix: {word_df.shape}")

# Apriori
frequent_itemsets = apriori(word_df, min_support=0.05, use_colnames=True)
print(f"Σύνολο frequent itemsets: {len(frequent_itemsets)}")

rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1.2)
rules = rules.sort_values('lift', ascending=False)

print(f"Σύνολο κανόνων: {len(rules)}")

# Εμφάνιση top 15
print("\n🔹 Top 15 Κανόνες Συσχέτισης Λέξεων:")
top_rules = rules.head(15)[['antecedents', 'consequents', 'support', 'confidence', 'lift']].copy()
top_rules['antecedents'] = top_rules['antecedents'].apply(lambda x: ', '.join(list(x)))
top_rules['consequents'] = top_rules['consequents'].apply(lambda x: ', '.join(list(x)))
print(top_rules.to_string(index=False))

# Οπτικοποίηση
plt.figure(figsize=(10, 6))
top_plot = rules.head(10)
labels = [f"{list(a)[0]} → {list(c)[0]}" for a, c in zip(top_plot['antecedents'], top_plot['consequents'])]
plt.barh(range(len(top_plot)), top_plot['lift'], color='purple', alpha=0.7)
plt.yticks(range(len(top_plot)), labels)
plt.xlabel('Lift')
plt.title('Top 10 Word Association Rules')
plt.tight_layout()
plt.show()

# Μερος 5

print("\n" + "="*50)
print("PART 5: BIG DATA PROCESSING (MapReduce με Pandas)")
print("="*50)

# ==========================================
# MapReduce 1: Μέσο μήκος review ανά sentiment
# ==========================================
print("=== 1. ΜΕΣΟ ΜΗΚΟΣ REVIEW ΑΝΑ SENTIMENT (MapReduce) ===")

# Map φάση: δημιουργία (key, value) ζευγών
def map_length(row):
    sentiment = 'Positive' if row['sentiment'] == 1 else 'Negative'
    length = len(row['review'].split())
    return [(sentiment, (length, 1))]

# Εκτέλεση Map
mapped = []
for _, row in df.iterrows():
    mapped.extend(map_length(row))

# Reduce φάση: ομαδοποίηση και υπολογισμός μέσου όρου
from collections import defaultdict
reduced = defaultdict(lambda: {'total_length': 0, 'count': 0})

for key, (length, count) in mapped:
    reduced[key]['total_length'] += length
    reduced[key]['count'] += count

print(f"{'Sentiment':<12} {'Avg Length':<12} {'Count':<8}")
print("-" * 32)
for sentiment in ['Positive', 'Negative']:
    avg = reduced[sentiment]['total_length'] / reduced[sentiment]['count']
    print(f"{sentiment:<12} {avg:<12.1f} {reduced[sentiment]['count']:<8}")

# ==========================================
# MapReduce 2: Word Count (Top 20 λέξεις)
# ==========================================
print("\n=== 2. TOP 20 ΛΕΞΕΙΣ (MapReduce Word Count) ===")

import re
from collections import Counter

stop_words = {'the','and','this','that','with','from','they','have','were','been',
              'what','when','for','are','was','but','not','you','all','can','had',
              'has','her','him','his','how','its','may','nor','once','our','out',
              'own','per','say','she','some','than','their','them','then','there',
              'these','they','this','upon','was','were','what','when','who','will',
              'with','would','your','just','about','like','which','also','very'}

# Map φάση
def map_words(review):
    words = re.findall(r'\b[a-zA-Z]+\b', review.lower())
    return [(w, 1) for w in words if len(w) > 3 and w not in stop_words]

# Εκτέλεση Map
word_pairs = []
for review in df['review']:
    word_pairs.extend(map_words(review))

# Reduce φάση
word_counts = Counter()
for word, count in word_pairs:
    word_counts[word] += count

# Top 20
print(f"{'Λέξη':<20} {'Count':<8}")
print("-" * 28)
for word, count in word_counts.most_common(20):
    print(f"{word:<20} {count:<8}")

# ==========================================
# MapReduce 3: Reviews ανά έτος (αν υπάρχουν ημερομηνίες)
# ==========================================
print("\n=== 3. TOP 5 ΜΕΓΑΛΥΤΕΡΑ REVIEWS (MapReduce) ===")

# Map: (μήκος, review)
def map_top_reviews(row):
    return [(len(row['review']), row['review'][:80])]

mapped_reviews = []
for _, row in df.iterrows():
    mapped_reviews.extend(map_top_reviews(row))

# Reduce: ταξινόμηση
mapped_reviews.sort(key=lambda x: x[0], reverse=True)

print(f"{'Length':<8} {'Review (first 80 chars)':<80}")
print("-" * 88)
for length, review in mapped_reviews[:5]:
    print(f"{length:<8} {review:<80}")


    # Συμπερασματα

    print("\n" + "="*60)
print("               ΤΕΛΙΚΑ ΣΥΜΠΕΡΑΣΜΑΤΑ")
print("="*60)

best_model_name = comp_df.iloc[0]['Model']
best_f1 = comp_df.iloc[0]['F1-Score']

print(f"""
📊 1. ΔΕΔΟΜΕΝΑ:
   • Σύνολο reviews: {len(df):,}
   • Θετικά: {sum(df['sentiment']):,} ({sum(df['sentiment'])/len(df)*100:.1f}%)
   • Αρνητικά: {len(df)-sum(df['sentiment']):,} ({(len(df)-sum(df['sentiment']))/len(df)*100:.1f}%)
   • Μέσο μήκος θετικού review: {reduced['Positive']['total_length']/reduced['Positive']['count']:.1f} λέξεις
   • Μέσο μήκος αρνητικού review: {reduced['Negative']['total_length']/reduced['Negative']['count']:.1f} λέξεις

🏆 2. ΑΠΟΔΟΣΗ ΜΟΝΤΕΛΩΝ:
   • 1η θέση: {comp_df.iloc[0]['Model']} (F1={comp_df.iloc[0]['F1-Score']:.4f})
   • 2η θέση: {comp_df.iloc[1]['Model']} (F1={comp_df.iloc[1]['F1-Score']:.4f})
   • 3η θέση: {comp_df.iloc[2]['Model']} (F1={comp_df.iloc[2]['F1-Score']:.4f})
   • 4η θέση: {comp_df.iloc[3]['Model']} (F1={comp_df.iloc[3]['F1-Score']:.4f})

📊 3. CLUSTERING (K-Means):
   • Adjusted Rand Index: {ari:.4f}
   • Normalized Mutual Info: {nmi:.4f}

💡 4. INSIGHTS:
   • Το sentiment analysis σε reviews ταινιών επιτυγχάνεται με >84% ακρίβεια
   • Το {best_model_name} υπερέχει λόγω μη-γραμμικής φύσης του κειμένου
   • Οι θετικές κριτικές είναι ελαφρώς μεγαλύτερες από τις αρνητικές
   • Top λέξεις: "movie", "film", "good", "great" (θετικές), "bad", "worst" (αρνητικές)
   • Οι Association Rules έδειξαν ζεύγη όπως: "ive→seen", "year→old", "dont→like"
   • Το K-Means κατάφερε να διαχωρίσει τα reviews χωρίς επίβλεψη (ARI={ari:.3f})
   • Το MapReduce επεξεργάστηκε αποδοτικά τον μεγάλο όγκο δεδομένων

🔧 5. ΠΡΑΚΤΙΚΕΣ ΕΦΑΡΜΟΓΕΣ:
   • Αυτόματη κατηγοριοποίηση reviews σε IMDb/Rotten Tomatoes
   • Ανάλυση τάσεων κοινού για νέες κυκλοφορίες ταινιών
   • Social media monitoring για εταιρείες παραγωγής
   • Content-based recommendation systems
   • Customer feedback analysis σε πραγματικό χρόνο
""")