import re

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)  # remove URLs
    text = re.sub(r"<[^>]+>", " ", text)                # remove HTML tags
    text = re.sub(r"[^a-z\s]", " ", text)               # keep letters only
    return re.sub(r"\s+", " ", text).strip()            # collapse extra spaces