import pandas as pd, joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from preprocess import clean_text

import os
os.makedirs("model", exist_ok=True)

# 1. Load the data and join headline + body
df = pd.read_csv("data/fake_or_real_news.csv").dropna(subset=["text"])
df["content"] = (df["title"].fillna("") + " " + df["text"]).apply(clean_text)

# 2. Split 80/20, keeping the same FAKE/REAL ratio in both parts
X_train, X_test, y_train, y_test = train_test_split(
    df["content"], df["label"], test_size=0.2, random_state=7, stratify=df["label"])

# 3. Build the pipeline: text -> TF-IDF numbers -> classifier
pipe = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", max_df=0.7,
                              ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
    ("clf", PassiveAggressiveClassifier(max_iter=1000, C=0.5, random_state=42)),
])

# 4. Train and evaluate
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred, labels=["FAKE", "REAL"]))

# 5. Save the model for the web app
joblib.dump(pipe, "model/pipeline.joblib")