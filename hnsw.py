import yaml
import json
import numpy as np
import hnswlib

from pathlib import Path
ROOT = Path(__file__).parent
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
EF = HNSW["EF_construct"]
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
    
    # Saving to config file
    config["HNSW"]["number of elements"] = num_elements
    config["HNSW"]["dim"] = dim
    with open(ROOT / "config.yaml", "w") as f:
        yaml.safe_dump(config, f)
    
    # Initialize HNSW index
    index = hnswlib.Index(space= SPACE, dim= dim)
    index.init_index(max_elements= 2*num_elements, M= M, ef_construction= EF)
    
    # Add embeddings to the index
    ids = np.arange(num_elements)
    index.add_items(embeddings, ids)
    
    # Save the index to disk
    index.save_index(str(HNSW_PATH))
    
    print(f"Saved index to {HNSW_PATH}")
    
def update(index: hnswlib.Index):
    """Updates an existing HNSW index with new embeddings."""
    # Load new embeddings and chunks
    with open(CHUNKS_PATH, 'r', encoding= 'utf-8') as f:
        chunks = json.load(f)
    embeddings = np.load(EMBED_PATH)
    
    assert embeddings.shape[0] == len(chunks), (f"Mismatch: {embeddings.shape[0]} embeddings vs {len(chunks)} chunks.")
    
    current_num_elements = index.get_current_count()
    new_embeddings = embeddings[current_num_elements:]
    num_new = new_embeddings.shape[0]
    
    # Check whether we exceed max num of elements
    if num_new == 0:
        print("No new embeddings to add.")
        return
    
    if current_num_elements + num_new >= index.get_max_elements():
        index.resize_index(2*(current_num_elements + num_new))
        
    # Arrange the ids and add embeddings to index
    new_ids = np.arange(current_num_elements, current_num_elements + num_new)
    index.add_items(new_embeddings, new_ids)
    
    # Save the index to disk
    index.save_index(str(HNSW_PATH))
    
    # Update config params
    config["HNSW"]["number of elements"] = index.get_current_count()
    with open(ROOT / "config.yaml", "w") as f:
        yaml.safe_dump(config, f)
    
    print(f"Added {num_new} new items. Index now holds {index.get_current_count()} total.")
    
if __name__ == "__main__":
    index_file_exists = HNSW_PATH.is_file() and HNSW_PATH.stat().st_size > 0

    if index_file_exists:
        index = hnswlib.Index(space=SPACE, dim=HNSW["dim"])
        index.load_index(str(HNSW_PATH))
        print("Updating tree")
        update(index)
    else:
        print("building tree")
        build_tree()