from flask import Flask, render_template, request, redirect, url_for
import joblib
import numpy as np
from preprocess import clean_text

app = Flask(__name__)
pipe = joblib.load("model/pipeline.joblib")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":          # someone opened /predict directly
        return redirect(url_for("home"))

    title = request.form.get("title", "").strip()
    text = request.form.get("text", "").strip()
    content = clean_text(title + " " + text)

    if len(content.split()) < 5:
        return render_template("index.html", error="Please paste at least a few sentences.",
                               title=title, text=text)

    label = pipe.predict([content])[0]

    # Confidence: turn the model's score into a 50-100% number
    score = pipe.decision_function([content])[0]
    confidence = round(100 / (1 + np.exp(-abs(score) * 2)))

    # Words in this article that pushed the model toward FAKE or REAL
    vec = pipe.named_steps["tfidf"]
    clf = pipe.named_steps["clf"]
    weights = vec.transform([content]).multiply(clf.coef_[0]).tocsr()
    words = vec.get_feature_names_out()
    pairs = sorted(zip(weights.data, weights.indices))
    fake_words = [words[i] for w, i in pairs[:6] if w < 0]
    real_words = [words[i] for w, i in reversed(pairs[-6:]) if w > 0]

    return render_template("result.html", label=label, confidence=confidence,
                           title=title, text=text[:400],
                           fake_words=fake_words, real_words=real_words)


if __name__ == "__main__":
    app.run(debug=True)