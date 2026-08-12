
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

def load_existing() -> tuple[list[dict], int]:
    """Load existing articles (if any) and return them plus the next free id."""
    if not OUT_PATH.exists():
        return [], 0

    with open(OUT_PATH, "r", encoding="utf-8") as f:
        existing = json.load(f)

    next_id = max((a["id"] for a in existing), default=-1) + 1
    return existing, next_id


def main() -> None:
    """Load new Wikipedia articles, clean them, and append to existing JSON."""
    existing_articles, next_id = load_existing()
    print(f"Found {len(existing_articles)} existing articles. New ids start at {next_id}.")

    # Rebuild seen_hashes from existing articles so we don't re-add duplicates
    seen_hashes = {
        hashlib.md5(a["text"][:200].encode("utf-8")).hexdigest()
        for a in existing_articles
    }

    ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)

    new_articles = []
    article_id = next_id

    for row in ds:
        if len(new_articles) >= N_ARTICLES:
            break

        title = row["title"].strip()
        text = clean_text(row["text"])

        if len(text.split()) < MIN_WORDS:
            continue

        fingerprint = hashlib.md5(text[:200].encode("utf-8")).hexdigest()
        if fingerprint in seen_hashes:
            continue
        seen_hashes.add(fingerprint)

        new_articles.append({"id": article_id, "title": title, "text": text})
        article_id += 1

        if len(new_articles) % 100 == 0:
            print(f"Loaded {len(new_articles)} new articles")

    print(f"Loaded {len(new_articles)} new articles in total")

    all_articles = existing_articles + new_articles
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_articles, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(all_articles)} total articles to {OUT_PATH}")
    
if __name__ == "__main__":
    main()
