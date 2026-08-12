import yaml
import json
import numpy as np
import hnswlib

from pathlib import Path
ROOT = Path(__file__).parent.parent

# -------- CONFIG --------
with open(ROOT / "config.yaml", 'r') as f:
    config = yaml.safe_load(f)

EMBED = config["Embedding"]
CHUNK = config["Chunking"]
HNSW = config["HNSW"]

EMBED_PATH = ROOT / EMBED["Savepath"]
CHUNKS_PATH = ROOT / CHUNK["Savepath"]
HNSW_PATH = ROOT / HNSW["Savepath"]

SPACE = HNSW["Distance Metric"]
M = HNSW["Max Edges"]
EF = HNSW["EF"]
# -------------------------


def build_tree() -> None:
    """Builds an HNSW index from precomputed embeddings and saves it to disk."""
    # Load embeddings and chunks
    with open(CHUNKS_PATH, 'r', encoding= 'utf-8') as f:
        chunks = json.load(f)
        
    embeddings = np.load(EMBED_PATH)
    assert embeddings.shape[0] == len(chunks), (f"Mismatch: {embeddings.shape[0]} embeddings vs {len(chunks)} chunks.")
    
    num_elements, dim = embeddings.shape
    print(f"Building HNSW index: {num_elements} vectors, dim={dim}")
    
    # Initialize HNSW index
    index = hnswlib.Index(space= SPACE, dim= dim)
    index.init_index(max_elements= num_elements, M= M, ef_construction= EF)
    
    # Add embeddings to the index
    ids = np.arange(num_elements)
    index.add_items(embeddings, ids)
    
    # Save the index to disk
    index.save_index(str(HNSW_PATH))
    
    print(f"Saved index to {HNSW_PATH}")
    
if __name__ == "__main__":
    build_tree()