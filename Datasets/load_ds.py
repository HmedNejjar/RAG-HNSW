
#type: ignore
from pathlib import Path
ROOT = Path(__file__).parent.parent

from datasets import load_dataset
import re
import hashlib
import json


# -------- CONFIG --------
N_ARTICLES = 9000          # how many articles to pull (streamed, so this is cheap)
MIN_WORDS = 200            # drop stubs
OUT_PATH = ROOT / "Datasets" / "wiki_articles.json"
# -------------------------


def clean_text(text: str) -> str:
    """Strip residual wiki-markup artifacts and normalize whitespace."""
    # remove reference markers like [1], [23]
    text = re.sub(r"\[\d+\]", "", text)
    # remove leftover section markers if any (== Header ==)
    text = re.sub(r"={2,}\s*.*?\s*={2,}", "", text)
    # collapse whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def main() -> None:
    """Load Wikipedia articles, clean them, and save to JSON."""
    ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
    
    seen_hashes = set()
    articles = []
    
    for row in ds:
        if len(articles) >= N_ARTICLES:
            break
        id = row["id"]
        title = row["title"].strip()
        text = clean_text(row["text"])
        
        if len(text.split()) < MIN_WORDS:
            continue
        
        fingerprint = hashlib.md5(text[:200].encode("utf-8")).hexdigest()
        if fingerprint in seen_hashes:
            continue
        seen_hashes.add(fingerprint)
        
        articles.append({"id": id, "title": title, "text": text})
        
        if len(articles) % 100 == 0:
            print(f"Loaded {len(articles)} articles")
        
    print(f"Loaded {len(articles)} articles in total")
    
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)
        
    print(f"Saved {len(articles)} articles to {OUT_PATH}")
    
if __name__ == "__main__":
    main()
