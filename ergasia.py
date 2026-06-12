import numpy as np
import os
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn as sns
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import kagglehub
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB      
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.feature_extraction.text import CountVectorizer
from mlxtend.frequent_patterns import apriori, association_rules
import re
from collections import Counter
import math
from sklearn import metrics
import time

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

#Kaggle API token
os.environ['KAGGLE_API_TOKEN'] = 'KGAT_d3d79d12cd6971f4edef5cb83dc2ba8d'
path = kagglehub.dataset_download("abdallahwagih/imdb-movie-reviews")
print(f"Dataset downloaded to: {path}")
aclpth = os.path.join(path, "aclImdb")

reviews = []
sentiments = []

# Load training reviews
for label, sentiment_value in [("pos", 1), ("neg", 0)]:
    fldpth = os.path.join(aclpth, "train", label)
    
    if os.path.exists(fldpth):
        for filename in os.listdir(fldpth):
            if filename.endswith(".txt"):
                flpth = os.path.join(fldpth, filename)
                with open(flpth, 'r', encoding='utf-8') as f:
                    reviews.append(f.read())
                    sentiments.append(sentiment_value)

dtfrm = pd.DataFrame({'review': reviews, 'sentiment': sentiments})
print(f"Φορτώθηκαν {len(dtfrm)} reviews")
print(f"Θετικά: {sum(dtfrm['sentiment'])}")
print(f"Αρνητικά: {len(dtfrm) - sum(dtfrm['sentiment'])}")
print(dtfrm.head())

#αφαιρεση αχρηστων χαρακτηρων και χαρακτηριστικων καθως και stopwords και ληματοποιηση
def cleanup(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and len(w) > 2]
    return ' '.join(tokens)

dtfrm['cleanreview'] = dtfrm['review'].apply(cleanup)


def prepareclust(dtfrm, text_column='review', label_column='sentiment', max_features=1000):
    #μετατροπη κειμένων σε κάτι κλιμακώσιμο
    X_text = dtfrm[text_column].values
    y = dtfrm[label_column].values if label_column in dtfrm.columns else None
    tfidf = TfidfVectorizer(max_features=max_features, stop_words='english')
    X_tfidf = tfidf.fit_transform(X_text).toarray()
    print(f"Αρχικό σχήμα τext data: {X_text.shape}")
    print(f"Έπειτα από επεξεργασία TF-IDF: {X_tfidf.shape}")
    
    #εφαρμογη κανονικοποιησης και pca
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_tfidf)
    n_components = min(50, X_scaled.shape[1])
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    print(f"Explained variance ratio: {pca.explained_variance_ratio_[:10].sum():.2%}")
    
    #δημιουργια dataframe για επιστροφη
    pcadtfrm = pd.DataFrame(data=X_pca[:, :2], columns=['PC1', 'PC2'])
    if y is not None:
        pcadtfrm['target'] = y
        pcadtfrm['targetname'] = ['Positive' if i == 1 else 'Negative' for i in y]
    return {
        'X_scaled': X_scaled,
        'X_pca': X_pca,
        'pcadtfrm': pcadtfrm,
        'vectorizer': tfidf,
        'scaler': scaler,
        'pca': pca,
        'feature_names': tfidf.get_feature_names_out()
    }


#kmeans
def kaymeans(printplots, pca, title, return_best=True):
    numberOfRows, numberOfColumns = pca.shape
    sses = []
    sils = []
    times = []
    avg = 0.0
    
    bestsil = -1
    bestk = 2
    bestlab = None
    bestwhole = None
    
    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=0)
        start = time.time()
        y_kmeans = kmeans.fit_predict(pca)
        ktime = time.time() - start
        avg += ktime
        times.append(ktime)
        IDX = kmeans.labels_
        C = kmeans.cluster_centers_
        
        # Calculate SSE
        sse_total = 0.0
        for i in range(k):
            for j in range(numberOfRows):
                if IDX[j] == i:
                    sse_total = sse_total + math.dist(pca[j], C[i])**2
        sses.append(sse_total)
        
        # Calculate Silhouette score
        sil = metrics.silhouette_score(pca, IDX)
        sils.append(sil)
        print(f"For {title} (kmeans) cluster k={k}, SSE={sse_total:.4f} & Silhouette={sil:.4f}")
        
        # Track best silhouette score
        if sil > bestsil:
            bestsil = sil
            bestk = k
            bestlab = y_kmeans.copy()
            bestwhole = kmeans

        if printplots == 1:
            plt.figure(k-1)
            plt.scatter(pca[:, 0], pca[:, 1], c=y_kmeans, cmap='viridis')
            plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', marker='X')
            plt.title(f'K-Means for k={k} clusters for {title} dataset')
            plt.show()
    
    avg /= 9.0
    
    if return_best:
        # Return best clusters along with metrics
        return {
            'sses': sses,
            'sils': sils,
            'times': times,
            'avg_time': avg,
            'bestk': bestk,
            'bestsil': bestsil,
            'bestlab': bestlab,
            'bestwhole': bestwhole,
            'all_labels': [kmeans.labels_ for k in range(2, 11)]  # Store all for reference
        }
    else:
        return sses, sils, times, avg

#kmeans
results = prepareclust(dtfrm, text_column='review', label_column='sentiment', max_features=2000)

print("\nRunning K-Means clustering...")
printplots = 0  #δεν μας ενδιαφερουν τα πλοτσ τοσο
kmeansres = kaymeans(printplots, results['X_pca'], 'IMDB Reviews', return_best=True)

bestclust = kmeansres['bestlab']
bestk = kmeansres['bestk']

print(f"\nΧρησιμοποιώντας το καλύτερο k={bestk} με Silhouette={kmeansres['bestsil']:.4f}")
results['pcadtfrm']['cluster'] = bestclust

# Visualize the clusters (using best k)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.scatterplot(data=results['pcadtfrm'], x='PC1', y='PC2', hue='targetname', alpha=0.6)
plt.title('Actual Sentiment Labels')

plt.subplot(1, 2, 2)
sns.scatterplot(data=results['pcadtfrm'], x='PC1', y='PC2', hue='cluster', alpha=0.6, palette='Set2')
plt.title(f'K-Means Clusters (k={bestk})')

plt.tight_layout()
plt.show()

# Calculate performance metrics
actual_labels = results['pcadtfrm']['target']
ari = adjusted_rand_score(actual_labels, bestclust)
nmi = normalized_mutual_info_score(actual_labels, bestclust)

print(f"\nΑποδοτικότητα Συσταδοποίησης (k={bestk}):")
print(f"Adjusted Rand Index: {ari:.3f}")
print(f"Normalized Mutual Info: {nmi:.3f}")




#Μερος 1
print("\n" + "="*50)
print("PART 1: Μοντέλα Κατηγοριοποιήσης")
print("="*50)

# Χρησιμοποιούμε το TF-IDF (όχι PCA) για classification
X = results['X_scaled']  
y = dtfrm['sentiment'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"Train size: {X_train.shape}")
print(f"Test size: {X_test.shape}")

models = {
    'Naive Bayes': GaussianNB(),           
    'Neural Network': MLPClassifier(
        hidden_layer_sizes=(128, 64), 
        max_iter=100, 
        early_stopping=True,
        random_state=42
    ),
    'Decision Tree': DecisionTreeClassifier(max_depth=20, random_state=42)
}

classification_results = {}
#εκπαίδευση και αξιολόγηση
for name, model in models.items():
    print(f"\nΕκπαίδευση {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_prob = model.decision_function(X_test)
    else:
        y_prob = None
    
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


#μερος 2

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

#πίνακες συγχησης
for i, (name, res) in enumerate(classification_results.items()):
    row, col = i // 2, i % 2
    
    cm = confusion_matrix(y_test, res['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[row, col],
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    axes[row, col].set_title(f'{name}\nAccuracy: {res["accuracy"]:.3f}')
    axes[row, col].set_xlabel('Predicted')
    axes[row, col].set_ylabel('Actual')

#roc καμπυλες
colors = ['blue', 'green', 'orange']
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


#μερος 3
print("\n" + "="*50)
print("PART 3: ΣΥΓΚΡΙΣΗ ΜΟΝΤΕΛΩΝ")
print("="*50)

#πινακες συγκρισης
compdata = []
for name, res in classification_results.items():
    compdata.append({
        'Model': name,
        'Accuracy': res['accuracy'],
        'Precision': res['precision'],
        'Recall': res['recall'],
        'F1-Score': res['f1']
    })

compdf = pd.DataFrame(compdata)
compdf = compdf.sort_values('F1-Score', ascending=False)

print("\n=== ΣΥΓΚΡΙΤΙΚΟΣ ΠΙΝΑΚΑΣ ===")
print(compdf.round(4).to_string(index=False))

#οπτικοποιηση
fig, ax = plt.subplots(figsize=(10, 4))
compdf.set_index('Model').plot(kind='bar', ax=ax, colormap='viridis')
plt.title('Σύγκριση Μοντέλων Classification - IMDB Reviews')
plt.xlabel('Μοντέλα')
plt.ylabel('Score')
plt.xticks(rotation=0)
plt.legend(loc='lower right')
plt.grid(axis='y', alpha=0.3)
for container in ax.containers:
    ax.bar_label(container, fmt='%.3f', fontsize=9)
plt.tight_layout()
plt.show()

#επιλογη πιο αποδοτικου μοντελου
bstmodl = compdf.iloc[0]
print(f"\nΚαλύτερο μοντέλο: {bstmodl['Model']}")
print(f"F1-Score: {bstmodl['F1-Score']:.4f}")


#μερος 4
print("\n" + "="*50)
print("PART 4: Κανόνες συσχέτισης - Ζευγάρια λέξεων")
print("="*50)

#τυχαια δειγματοληψεια
np.random.seed(42)
smplidx = np.random.choice(len(dtfrm), size=min(3000, len(dtfrm)), replace=False)
dftmp = dtfrm.iloc[smplidx]

#δυαδικός πινακας με τις top 80 λέξεις
cv = CountVectorizer(max_features=80, binary=True, stop_words='english')
wordarr = cv.fit_transform(dftmp['cleanreview'])
wrddf = pd.DataFrame(wordarr.toarray(), columns=cv.get_feature_names_out())

print(f"Διαστάσεις δυαδικού πίνακα: {wrddf.shape}")

#συνολα apriori
itemsets = apriori(wrddf, min_support=0.05, use_colnames=True)
print(f"Σύνολο συχνών ζευγαριών: {len(itemsets)}")

rules = association_rules(itemsets, metric='lift', min_threshold=1.2)
rules = rules.sort_values('lift', ascending=False)

def removedoubles(rules_df):
    seen = set()
    mask = []
    
    for idx, row in rules_df.iterrows():
        #συγχωνευση διπλωτυπων
        ant = frozenset(row['antecedents'])
        con = frozenset(row['consequents'])
        combined = ant.union(con)
        if combined not in seen:
            seen.add(combined)
            mask.append(True)
        else:
            mask.append(False)
    return rules_df[mask].reset_index(drop=True)
    
rules_unique = removedoubles(rules)
print(f"Σύνολο αρχικών κανόνων: {len(rules)}")
print(f"Σύνολο ΜΟΝΑΔΙΚΩΝ κανόνων: {len(rules_unique)}")

#οπτικοποίηση κορυφαίων ζεύγων
print("\nΠρώτοι 15 κανόνες συσχέτισης λέξεων:")
tophits = rules_unique.head(15)[['antecedents', 'consequents', 'support', 'confidence', 'lift']].copy()
tophits['antecedents'] = tophits['antecedents'].apply(lambda x: ', '.join(sorted(list(x))))
tophits['consequents'] = tophits['consequents'].apply(lambda x: ', '.join(sorted(list(x))))

print(tophits.to_string(index=False))
plt.figure(figsize=(12, 6))
top_plot = rules_unique.head(10)
labels = []
for a, c in zip(top_plot['antecedents'], top_plot['consequents']):
    ant_str = ', '.join(sorted(list(a)))
    con_str = ', '.join(sorted(list(c)))
    labels.append(f"{ant_str} → {con_str}")
plt.barh(range(len(top_plot)), top_plot['lift'], color='purple', alpha=0.7)
plt.yticks(range(len(top_plot)), labels)
plt.xlabel('Lift')
plt.title('Πρώτοι 10 μοναδικοί κανόνες συσχέτισης λέξεων')
plt.tight_layout()
plt.show()

#μερος 5
print("\n" + "="*50)
print("PART 5: Διαχείρηση δεδομένων μεγάλης κλίμακας")
print("="*50)

#μέσο μήκος κριτκής ανά κατηγορία(καλή/κακή)

#δημιουργία χάρτη
def maplen(row):
    sentiment = 'Positive' if row['sentiment'] == 1 else 'Negative'
    length = len(row['review'].split())
    return [(sentiment, (length, 1))]

mapped = []
for _, row in dtfrm.iterrows():
    mapped.extend(maplen(row))

#φάση μείωσης
from collections import defaultdict
reduced = defaultdict(lambda: {'total_length': 0, 'count': 0})

for key, (length, count) in mapped:
    reduced[key]['total_length'] += length
    reduced[key]['count'] += count

#εκτύπωση
print(f"{'Sentiment':<12} {'Avg Length':<12} {'Count':<8}")
print("-" * 32)
for sentiment in ['Positive', 'Negative']:
    avg = reduced[sentiment]['total_length'] / reduced[sentiment]['count']
    print(f"{sentiment:<12} {avg:<12.1f} {reduced[sentiment]['count']:<8}")


#ευρεση 20 πιο συχνων λεξεων

#δημιουργία χάρτη
def mapwrds(review):
    words = re.findall(r'\b[a-zA-Z]+\b', review.lower())
    return [(w, 1) for w in words if len(w) > 3 and w not in stop_words]
pairs = []
for review in dtfrm['review']:
    pairs.extend(mapwrds(review))

#φάση μείωσης
word_counts = Counter()
for word, count in pairs:
    word_counts[word] += count

#εκτύπωση
print(f"{'Λέξη':<20} {'Count':<8}")
print("-" * 28)
for word, count in word_counts.most_common(20):
    print(f"{word:<20} {count:<8}")

#μεγαλύτερες κριτηκές

#δημιουργία χάρτη
def maplenrev(row):
    return [(len(row['cleanreview']), row['cleanreview'][:80])]

mappedrev = []
for _, row in dtfrm.iterrows():
    mappedrev.extend(maplenrev(row))

#φάση μείωσης/ταξινόμισης
mappedrev.sort(key=lambda x: x[0], reverse=True)

print(f"{'Length':<8} {'Review (first 80 chars)':<80}")
print("-" * 88)
for length, review in mappedrev[:5]:
    print(f"{length:<8} {review:<80}")




print("\n" + "="*60)
print("               ΤΕΛΙΚΑ ΣΥΜΠΕΡΑΣΜΑΤΑ")
print("="*60)

best = compdf.iloc[0]['Model']
bestf1 = compdf.iloc[0]['F1-Score']

print(f"""
Μερικά γενικά στοιχεία για τα δεδομένα:
Σύνολο reviews: {len(dtfrm):,}
Θετικά: {sum(dtfrm['sentiment']):,} ({sum(dtfrm['sentiment'])/len(dtfrm)*100:.1f}%)
Αρνητικά: {len(dtfrm)-sum(dtfrm['sentiment']):,} ({(len(dtfrm)-sum(dtfrm['sentiment']))/len(dtfrm)*100:.1f}%)
Μέσο μήκος θετικού review: {reduced['Positive']['total_length']/reduced['Positive']['count']:.1f} λέξεις
Μέσο μήκος αρνητικού review: {reduced['Negative']['total_length']/reduced['Negative']['count']:.1f} λέξεις

Σύγκριση μοντέλων βάση βαθμολογίας f1 (βάθρο):
1η θέση: {compdf.iloc[0]['Model']} (F1={compdf.iloc[0]['F1-Score']:.4f})
2η θέση: {compdf.iloc[1]['Model']} (F1={compdf.iloc[1]['F1-Score']:.4f})
3η θέση: {compdf.iloc[2]['Model']} (F1={compdf.iloc[2]['F1-Score']:.4f})

Απόδοση συσταδοποίησης kmeans:
Adjusted Rand Index: {ari:.4f}
Normalized Mutual Info: {nmi:.4f}
""")