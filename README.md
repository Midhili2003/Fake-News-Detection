# TruthLens: Fake News Detection with Machine Learning

TruthLens is a web app that predicts whether a news article is **real or fake** from its language,
plus a study of how far such models can be trusted on news they have never seen.

The project goes beyond a single accuracy number:

- Compares four machine-learning classifiers.
- Tests them on a second, independent dataset.
- Finds and removes data leakage.
- Fine-tunes a **DistilBERT** transformer to see whether reading words in context generalizes better than counting them.

![Home page](screenshots/home.png)

| Fake-style article | Real-style article |
|---|---|
| ![Fake result](screenshots/result_fake.png) | ![Real result](screenshots/result_real.png) |

---

## Highlights

- **95% accuracy** on the training dataset with TF-IDF + Passive Aggressive Classifier.
- **But only about 70% on unseen data.** A cross-dataset test showed that in-dataset accuracy overstates real-world performance.
- **DistilBERT improved unseen-data accuracy from 68% to 82%** when trained on the same data.
- **More data didn't fix it; variety did.** 7× more training data barely helped. Error analysis showed the larger dataset's "real" class was mostly one outlet (Reuters), so the model learned that outlet's style as the definition of "real."
- **Data leakage found and removed.** Outlet names, datelines and social-media links were letting models take shortcuts.

---

## Features

- Paste a headline and article text, and get a **FAKE / REAL** verdict with a confidence score.
- See the **words that pushed the prediction** toward fake or real.
- Try **built-in sample articles** with one click.
- Two-page interface (detector + result) built with Flask, HTML and CSS.

## Tech stack

| Area | Tools |
|---|---|
| Language | Python 3.10+ |
| Machine learning | scikit-learn (TF-IDF, Passive Aggressive, Logistic Regression, Linear SVM, Naive Bayes) |
| Deep learning | PyTorch, Hugging Face Transformers (DistilBERT), Google Colab (T4 GPU) |
| Data | pandas, NumPy |
| Web app | Flask, HTML, CSS, JavaScript |

---

## Getting started

These steps run the web app on your own computer. It takes about 10 minutes.

### 1. Prerequisites

- **Python 3.10 or newer:** [python.org/downloads](https://www.python.org/downloads/). On Windows, tick **"Add Python to PATH"** during installation.
- **Git:** [git-scm.com](https://git-scm.com/downloads).

Check both are installed:

```bash
python --version
git --version
```

On Mac/Linux, use `python3` instead of `python` if the command isn't found.

### 2. Download the project

```bash
git clone https://github.com/<your-username>/fake-news-detection.git
cd fake-news-detection
```

### 3. Create a virtual environment and install the libraries

**Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**Mac / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

You'll see `(.venv)` at the start of your terminal line when the environment is active.

### 4. Download the dataset

The dataset isn't included in this repository because of its size.

1. Download **fake_or_real_news.csv** from [Kaggle](https://www.kaggle.com/datasets/hassanamin/textdb3) or the [GitHub mirror](https://raw.githubusercontent.com/lutzhamel/fake-news/master/data/fake_or_real_news.csv).
2. Create a folder named `data` in the project and put the file there:

```
fake-news-detection/
└── data/
    └── fake_or_real_news.csv
```

### 5. Train the model

```bash
python train.py
```

This takes under a minute. It prints the accuracy and saves the trained model to `model/pipeline.joblib`.

### 6. Run the web app

```bash
python app.py
```

Open **http://127.0.0.1:5000** in your browser. Click **Fake-style** or **Real-style** to try a sample, or paste your own article. Press `Ctrl+C` in the terminal to stop the server.

### Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError` | The virtual environment isn't active. Run the activate command from step 3 again. |
| `FileNotFoundError: data/fake_or_real_news.csv` | Check that the file is inside the `data` folder with exactly that name. |
| `FileNotFoundError: model/pipeline.joblib` | Run `python train.py` before `python app.py`. If the `model` folder is missing, create it first. |
| "running scripts is disabled" (Windows) | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then activate again. |
| Page looks unstyled or outdated | Press `Ctrl+F5` in the browser to clear its saved copy. |

---

## Reproducing the experiments (optional)

These steps reproduce the model comparison, cross-dataset test and DistilBERT results.

**1. Download the second dataset.** Get **WELFake** from [Kaggle](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification) and place `WELFake_Dataset.csv` in the `data` folder.

**2. Run the scripts in order:**

```bash
python compare_models.py        # compares 4 classifiers, saves model_comparison.csv
python prepare_welfake.py       # cleans WELFake, saves data/welfake_clean.csv
python cross_dataset_test.py    # cross-dataset test, saves cross_dataset_results.csv (5–15 min)
```

**3. Run the DistilBERT notebook.** Open `distilbert_fake_news.ipynb` in [Google Colab](https://colab.research.google.com):

1. Select **Runtime → Change runtime type → T4 GPU**.
2. Upload `fake_or_real_news.csv` and `welfake_clean.csv` to a Google Drive folder named `fake-news`.
3. Run the cells in order. It takes about 20–30 minutes in total.

---

## Results

### 1. Model comparison (original dataset, 80/20 split)

| Model | Accuracy | F1 (FAKE) |
|---|---|---|
| Passive Aggressive | 95.4% | 0.955 |
| Linear SVM | 95.0% | 0.951 |
| Logistic Regression | 94.8% | 0.948 |
| Naive Bayes | 91.3% | 0.910 |

The three linear models are within 1 point of each other, so the features matter more than the choice of classifier.

### 2. Cross-dataset test: does it work on news it hasn't seen?

| Trained on → Tested on | Passive Aggressive | Logistic Regression | Linear SVM | Naive Bayes |
|---|---|---|---|---|
| Original → Original (same) | 95.5% | 94.9% | 95.2% | 91.2% |
| Original → WELFake (new) | 68.2% | 68.4% | 67.8% | 63.0% |
| WELFake → WELFake (same) | 96.2% | 95.8% | 96.2% | 89.9% |
| WELFake → Original (new) | 70.8% | 71.8% | 71.4% | 64.2% |

Accuracy drops about 25 points on a new dataset: the models partly learn patterns specific to their own dataset.

### 3. Removing data leakage (WELFake-trained Logistic Regression)

| Cleaning round | Same dataset | New dataset |
|---|---|---|
| Basic cleanup | 97.1% | 69.9% |
| + outlet names in headlines removed | 96.5% | 72.1% |
| + links, sign-offs, overlap fix | 95.8% | 71.8% |

Removing shortcuts lowered same-dataset accuracy but raised new-data accuracy. The higher first number was partly inflated.

### 4. DistilBERT vs. TF-IDF

| Trained on → Tested on | TF-IDF (best) | DistilBERT |
|---|---|---|
| Original → Original (same) | 95.5% | 96.2% |
| Original → WELFake (new) | 68.4% | **82.4%** |
| WELFake → WELFake (same) | 96.2% | 99.5% |
| WELFake → Original (new) | 71.8% | 70.0% |

### 5. Error analysis: WELFake-trained DistilBERT on the original dataset

|  | Predicted FAKE | Predicted REAL |
|---|---|---|
| **Actually FAKE** | 3,127 | 37 |
| **Actually REAL** | 1,866 | 1,305 |

The model catches 99% of fake articles but labels **59% of real articles as fake**. WELFake's real class is 62% Reuters, so the model learned wire-news style as the definition of "real" and flags analysis or feature writing from other outlets as fake. Removing an outlet's name doesn't remove its writing style.

---

## How the data was cleaned

- **Text:** lowercased; URLs, HTML, numbers and punctuation removed; headline joined to article body.
- **Features (TF-IDF):** English stop words removed; words in more than 70% of articles, or in only one, dropped; single words and two-word phrases used.
- **WELFake:**
  - removed 9,693 duplicate or empty rows
  - worked out from the data which label means real
  - removed 6,174 articles that also appear in the original dataset, so the cross-dataset test is fair
  - stripped source giveaways (outlet names, datelines, links, social-media sign-offs) to prevent data leakage

## Project structure

```
fake-news-detection/
├── app.py                       # Flask web app
├── train.py                     # trains the TF-IDF + Passive Aggressive model used by the app
├── preprocess.py                # text cleaning shared by training and the app
├── compare_models.py            # compares 4 classifiers
├── prepare_welfake.py           # cleans the WELFake dataset
├── cross_dataset_test.py        # cross-dataset evaluation
├── explore.ipynb                # initial data exploration
├── distilbert_fake_news.ipynb   # DistilBERT fine-tuning (Google Colab)
├── model_comparison.csv         # results: model comparison
├── cross_dataset_results.csv    # results: cross-dataset test
├── distilbert_results.csv       # results: DistilBERT
├── templates/                   # HTML pages (base, index, result)
├── static/style.css             # styling
├── screenshots/                 # images used in this README
└── requirements.txt
```

## Datasets

| Dataset | Size | Source |
|---|---|---|
| fake_or_real_news | 6,335 articles, 2016 U.S. politics | [Kaggle](https://www.kaggle.com/datasets/hassanamin/textdb3) · [GitHub mirror](https://github.com/lutzhamel/fake-news) |
| WELFake | 72,134 articles from 4 merged datasets | [Kaggle](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification) |

## Limitations

- The models judge **writing style and topic, not factual truth**. A well-written false story can look real.
- Both datasets focus on 2016–2017 U.S. politics, so accuracy on other topics or recent news is likely lower.
- DistilBERT reads only the first ~180 words of each article, and results come from single training runs.
- Some outlet-specific style remains in WELFake (for example "Mr."/"Ms." from New York Times house style).

## Future work

- Use the DistilBERT model in the web app.
- Train on more varied sources, especially for the real-news class.
- Show related fact-checks using the Google Fact Check Tools API.

---

## Author

**Midhili Asok**, MS in Computer Science, New York University

[LinkedIn](https://www.linkedin.com/in/<your-linkedin>) · [GitHub](https://github.com/<your-username>)
