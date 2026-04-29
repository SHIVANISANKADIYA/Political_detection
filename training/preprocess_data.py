# preprocess_data.py
import json
import re
import nltk
from bs4 import BeautifulSoup
from pathlib import Path

# अगर punkt मौजूद न हो तो डाउनलोड कर लेगा (पहली बार)
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

# इनपुट/आउटपुट पाथ — आप चाहें तो बदल लें
input_file = Path("raw_scraped_political_news_articles")
output_file = Path("processed_political_news_articles.jsonl")

def clean_text(text: str) -> str:
    if not text:
        return ""
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r"\s+", " ", text)
    return text.strip()

processed_count = 0
with input_file.open("r", encoding="utf-8") as f_in, output_file.open("w", encoding="utf-8") as f_out:
    for line in f_in:
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            # skip invalid json lines
            continue

        title = clean_text(entry.get("title", "") or "")
        body = clean_text(entry.get("body", "") or "")
        full_text = f"{title}. {body}" if title else body
        if not full_text:
            continue

        # sentence segmentation (nltk punkt)
        sentences = nltk.sent_tokenize(full_text)

        # Write one JSON object per sentence so it's easy to annotate later
        for sent in sentences:
            obj = {"sentence": sent}
            f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")
            processed_count += 1

print(f"Done. Wrote {processed_count} sentences to {output_file}")