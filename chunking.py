def recursive_chunking(text: str, chunk_size: int, overlap_size: int) -> list[str]:
    """Recursively chunk a text into smaller chunks using a separator hierarchy."""

    # Clean and validate input
    text = text.strip()
    if not text:
        return []

    # Separators are tried from largest semantic boundary to smallest
    separators = ["\n\n", "\n", ". ", "! ", "? ", "; ", ": ", ", ", " "]

    # Recursively split the text, then merge pieces and add overlap
    good_splits = split_text(text, chunk_size, separators)
    return overlap_merging(good_splits, chunk_size, overlap_size)


def split_text(text: str, chunk_size: int, separators: list[str]) -> list[str]:

    # Base case: the entire text already fits inside the chunk size
    if len(text.split()) <= chunk_size:
        return [text] if text else []

    # Find the first separator that exists in the current text
    separator: str | None = None
    remaining_separators = separators

    for i, sep in enumerate(separators):
        if sep in text:
            separator = sep

            # Recursive calls only use smaller separators
            remaining_separators = separators[i + 1:]
            break

    # No separators remain: fall back to hard word-based splitting
    if separator is None:
        words = text.split()

        word_chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
        return word_chunks

    # Split the text using the current separator
    good_splits = []

    for split in text.split(separator):
        split = split.strip()

        if not split:
            continue

        # Keep pieces that already fit
        if len(split.split()) <= chunk_size:
            good_splits.append(split)

        else:
            # Piece is still too large, so recurse using smaller separators
            splits = split_text(split, chunk_size, remaining_separators)
            good_splits.extend(splits)

    return good_splits


def overlap_merging(good_splits: list[str], chunk_size: int, overlap_size: int) -> list[str]:
    """Greedily pack pieces into chunks up to chunk_size, then add overlap."""

    if not good_splits:
        return []

    # Greedily merge small pieces until adding another would exceed chunk_size
    chunks = []
    current = good_splits[0]

    for split in good_splits[1:]:
        candidate = current + "; " + split

        if len(candidate.split()) <= chunk_size:
            current = candidate
        else:
            # Current chunk is full, so save it and start a new one
            chunks.append(current)
            current = split

    # Save the final accumulated chunk
    chunks.append(current)

    # No overlap requested, or there is only one chunk
    if overlap_size <= 0 or len(chunks) < 2:
        return chunks

    # Add the end of the previous chunk to the beginning of each next chunk
    overlapped = [chunks[0]]

    for i in range(1, len(chunks)):
        prev_chunk = chunks[i - 1].split()

        # Take the last `overlap_size` words from the previous chunk
        overlap_txt = " ".join(prev_chunk[-overlap_size:])

        # Prepend the overlap to the current chunk
        overlapped.append(overlap_txt + " " + chunks[i])

    return overlapped

if __name__ == "__main__":
    import json
    import yaml
    from pathlib import Path
    
    ROOT = Path(__file__).parent
    
    # Load the chunking config from config.yaml
    with open(ROOT / "config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    ARTICLES = ROOT / "Datasets" / "wiki_articles.json"
    CHUNKS = ROOT / config["Chunking"]["Savepath"]
    
        
    CHUNK_SIZE = config["Chunking"]["Chunk size"]
    OVERLAP = config["Chunking"]["Overlap"]
    
    # Load the articles from the JSON file
    with open(ARTICLES, "r", encoding="utf-8") as f:
        articles = json.load(f)
        
    print(f"Loaded {len(articles)} articles.")
    
    # Chunk the articles and save to a new JSON file
    chunks = []
    
    for article in articles:
        splits = recursive_chunking(article["text"], CHUNK_SIZE, OVERLAP)
        
        for i, split in enumerate(splits):
            chunks.append({
                "chunk_id": f"{article["id"]}_{i}",
                "article_id": article["id"],
                "title": article["title"],
                "chunk_index": i,
                "text": split
            })
            
    print(f"Produced {len(chunks)} chunks from {len(articles)} articles "
          f"(avg {len(chunks) / len(articles):.1f} chunks/article).")
    
    with open(CHUNKS, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
        
    print(f"Saved to {CHUNKS}")