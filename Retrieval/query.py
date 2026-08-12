import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml
import json
import hnswlib
from Embedding import embed

# -------- CONFIG --------
with open(ROOT / "config.yaml", 'r') as f:
    config = yaml.safe_load(f)

EMBED = config["Embedding"]
CHUNK = config["Chunking"]
HNSW = config["HNSW"]

MODEL = EMBED["Model"]

EMBED_PATH = ROOT / EMBED["Savepath"]
CHUNKS_PATH = ROOT / CHUNK["Savepath"]
HNSW_PATH = ROOT / HNSW["Savepath"]

SPACE = HNSW["Distance Metric"]
K = HNSW["top_k"]
EF = HNSW["EF_search"]
NUM_ELEMENTS = HNSW["number of elements"]
DIM = HNSW["dim"]
# -------------------------

def retrieve(query: str, num_elements: int, dim: int) -> list:
    # Embed the query
    query = "Represent this sentence for searching relevant passages: " + query
    query_embed = embed([query], MODEL, 1)
    
    # Load the HNSW index
    index = hnswlib.Index(space= SPACE, dim= dim)
    index.load_index(str(HNSW_PATH), max_elements= num_elements)
    
    # Set the ef parameter for searching
    index.set_ef(EF)
    
    # Perform the search
    ids, distances = index.knn_query(query_embed, k= K)
    ids = ids[0]
    distances = distances[0]
    
    # Load the chunks
    with open(CHUNKS_PATH, 'r', encoding= 'utf-8') as f:
        chunks = json.load(f)
    
    # Save the results
    results = []
    for chunk_id, dist in zip(ids, distances):
        chunk = chunks[chunk_id]
        
        results.append({
            "score": float(1 - dist),
            "distance": dist,
            "chunk id": chunk_id,
            "article id": chunk["article_id"],
            "title": chunk["title"],
            "text": chunk["text"]
        })
        
    return results

if __name__ == "__main__":
    query = input("Enter a query: ")
    results = retrieve(query, NUM_ELEMENTS, DIM)
    
    for result in results:
        print(f"Score: {result['score']:.4f}, Distance: {result['distance']:.4f}, Chunk ID: {result['chunk id']}, Article ID: {result['article id']}, Title: {result['title']}")
        print(f"Text: {result['text'][:200]}\n")