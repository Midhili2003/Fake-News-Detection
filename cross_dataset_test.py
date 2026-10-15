import time
import warnings
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier, LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score
from preprocess import clean_text

warnings.filterwarnings("ignore")


def load(path):
    df = pd.read_csv(path).dropna(subset=["text"])
    df["content"] = (df["title"].fillna("") + " " + df["text"]).apply(clean_text)
    return df["content"], df["label"]


def make_models():
    return {
        "Passive Aggressive": PassiveAggressiveClassifier(max_iter=1000, C=0.5, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, C=10),
        "Linear SVM": LinearSVC(C=1.0),
        "Naive Bayes": MultinomialNB(alpha=0.1),
    }


def run(train_name, X_train, y_train, tests):
    """Train every model on one dataset, then score it on each test set."""
    tfidf = TfidfVectorizer(stop_words="english", max_df=0.7, min_df=2,
                            ngram_range=(1, 2), sublinear_tf=True, max_features=200_000)
    X_train_vec = tfidf.fit_transform(X_train)
    rows = []
    for name, model in make_models().items():
        start = time.time()
        model.fit(X_train_vec, y_train)
        for test_name, (X_test, y_test) in tests.items():
            y_pred = model.predict(tfidf.transform(X_test))
            rows.append({
                "Trained on": train_name,
                "Tested on": test_name,
                "Model": name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Macro F1": f1_score(y_test, y_pred, average="macro"),
            })
        print(f"  {name} done ({time.time() - start:.1f}s)")
    return rows, tfidf, model


# Load both datasets
X_orig, y_orig = load("data/fake_or_real_news.csv")
X_wel, y_wel = load("data/welfake_clean.csv")
print(f"Original: {len(X_orig):,} articles | WELFake: {len(X_wel):,} articles\n")

# Hold out 20% of each dataset for same-dataset testing
Xo_tr, Xo_te, yo_tr, yo_te = train_test_split(X_orig, y_orig, test_size=0.2, random_state=7, stratify=y_orig)
Xw_tr, Xw_te, yw_tr, yw_te = train_test_split(X_wel, y_wel, test_size=0.2, random_state=7, stratify=y_wel)

results = []

print("Experiment 1: train on ORIGINAL")
rows, _, _ = run("Original", Xo_tr, yo_tr, {
    "Original (same dataset)": (Xo_te, yo_te),
    "WELFake (new dataset)": (X_wel, y_wel),
})
results += rows

print("\nExperiment 2: train on WELFAKE")
rows, tfidf_w, _ = run("WELFake", Xw_tr, yw_tr, {
    "WELFake (same dataset)": (Xw_te, yw_te),
    "Original (new dataset)": (X_orig, y_orig),
})
results += rows

# Results table
table = pd.DataFrame(results)
pd.set_option("display.width", 200)
print("\n" + table.to_string(index=False, float_format="%.3f"))
table.to_csv("cross_dataset_results.csv", index=False)
print("\nSaved to cross_dataset_results.csv")

# Leakage check: which words does a WELFake-trained model rely on most?
lr = LogisticRegression(max_iter=1000, C=10).fit(tfidf_w.transform(Xw_tr), yw_tr)
words = tfidf_w.get_feature_names_out()
order = lr.coef_[0].argsort()          # negative = FAKE, positive = REAL
print("\nTop FAKE words (WELFake model):", ", ".join(words[order[:15]]))
print("Top REAL words (WELFake model):", ", ".join(words[order[-15:][::-1]]))