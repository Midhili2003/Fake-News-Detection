import time
import warnings
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier, LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from preprocess import clean_text

warnings.filterwarnings("ignore")  # hide the PassiveAggressive deprecation warning

# 1. Load data (same as train.py)
df = pd.read_csv("data/fake_or_real_news.csv").dropna(subset=["text"])
df["content"] = (df["title"].fillna("") + " " + df["text"]).apply(clean_text)

# 2. Same split as train.py, so results are comparable
X_train, X_test, y_train, y_test = train_test_split(
    df["content"], df["label"], test_size=0.2, random_state=7, stratify=df["label"])

# 3. Turn text into TF-IDF numbers once, and reuse for every model
tfidf = TfidfVectorizer(stop_words="english", max_df=0.7,
                        ngram_range=(1, 2), min_df=2, sublinear_tf=True)
X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)

# 4. The models to compare
models = {
    "Passive Aggressive": PassiveAggressiveClassifier(max_iter=1000, C=0.5, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, C=10),
    "Linear SVM": LinearSVC(C=1.0),
    "Naive Bayes": MultinomialNB(alpha=0.1),
}

# 5. Train and score each one
results = []
for name, model in models.items():
    start = time.time()
    model.fit(X_train_vec, y_train)
    train_time = time.time() - start

    y_pred = model.predict(X_test_vec)
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision (FAKE)": precision_score(y_test, y_pred, pos_label="FAKE"),
        "Recall (FAKE)": recall_score(y_test, y_pred, pos_label="FAKE"),
        "F1 (FAKE)": f1_score(y_test, y_pred, pos_label="FAKE"),
        "Train time (s)": train_time,
    })
    print(f"Done: {name}")

# 6. Show and save the comparison table
table = pd.DataFrame(results).sort_values("Accuracy", ascending=False)
print("\n" + table.to_string(index=False, float_format="%.4f"))
table.to_csv("model_comparison.csv", index=False)
print("\nSaved to model_comparison.csv")