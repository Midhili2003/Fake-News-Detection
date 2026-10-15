import re
import pandas as pd
from preprocess import clean_text

# 1. Load WELFake
df = pd.read_csv("data/WELFake_Dataset.csv")
print("Raw rows:", len(df))

df = df[["title", "text", "label"]]
df["title"] = df["title"].fillna("")
df = df.dropna(subset=["text"])
df = df[df["text"].str.strip().str.len() > 50]
df = df.drop_duplicates(subset=["text"])
print("After removing empty + duplicate articles:", len(df))

# 2. Work out which number means REAL (real articles mention Reuters far more often)
reuters_share = df.groupby("label")["text"].apply(lambda s: s.str.contains("Reuters").mean())
print("\nShare of articles mentioning 'Reuters' per label:\n", reuters_share)
real_value = reuters_share.idxmax()
df["label"] = df["label"].apply(lambda v: "REAL" if v == real_value else "FAKE")
print(f"-> label {real_value} = REAL")

# 3. Remove overlap with the original dataset FIRST (before the text is changed)
original = pd.read_csv("data/fake_or_real_news.csv")
fingerprint = lambda t: clean_text(t)[:200]
seen = set(original["text"].dropna().apply(fingerprint))
before = len(df)
df = df[~df["text"].apply(fingerprint).isin(seen)]
print(f"\nRemoved {before - len(df)} articles that overlap with fake_or_real_news.csv")

# 4. Remove giveaways that reveal the source instead of the content
def remove_giveaways(text):
    text = re.sub(r"^.{0,120}?\(Reuters\)\s*[-–—]\s*", "", text)                        # "WASHINGTON (Reuters) - "
    text = re.sub(r"\bhttps?\S*|\bwww\.\S+|pic\.twitter\.com\S*|\btwitter\.com\S*|@\w+",
                  " ", text, flags=re.IGNORECASE)                                       # links, tweet embeds, @handles
    text = re.sub(r"\b(Reuters|Breitbart( News)?|(The )?New York Times|NYTimes|NYT)\b",
                  " ", text, flags=re.IGNORECASE)                                       # outlet names
    text = re.sub(r"Featured image.*|Getty Images|21st Century Wire",
                  " ", text, flags=re.IGNORECASE)                                       # fake-site boilerplate
    text = re.sub(r"Follow (?:[A-Z][\w.]*\s?){1,3}on Twitter\b.*", " ", text)          # "Follow Milo on Twitter @..."
    return text

df["text"] = df["text"].apply(remove_giveaways)
df["title"] = df["title"].apply(remove_giveaways)

# 5. Save
print("\nFinal rows:", len(df))
print(df["label"].value_counts())
df.to_csv("data/welfake_clean.csv", index=False)
print("\nSaved to data/welfake_clean.csv")

for label in ["FAKE", "REAL"]:
    row = df[df["label"] == label].sample(1, random_state=3).iloc[0]
    print(f"\n===== Example {label} =====\n{row['title']}\n{row['text'][:300]}...")