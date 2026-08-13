import sys
from pathlib import Path
from typing import Literal

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml
import json
import hnswlib
from sentence_transformers import SentenceTransformer
from hnsw import update

HNSW_SPACE = Literal["l2", "ip", "cosine"]

class Retriever:
    def __init__(self, model_name: str, chunks: list[dict], HNSW_savepath: str | Path , space: HNSW_SPACE, ef_search: int,dim: int , top_k: int) -> None:
        if space not in ("l2", "ip", "cosine"):
            raise ValueError(f"Unsupported HNSW space: {space!r}. Expected one of: 'l2', 'ip', 'cosine'.")

        self.model = SentenceTransformer(model_name)
        self.chunks = chunks
        self.top_k = top_k

        # Load the HNSW index and set the EF
        
        self.index = hnswlib.Index(space=space, dim=dim)
        self.index.load_index(str(HNSW_savepath))
        update(self.index)
        self.index.set_ef(ef_search)
        
    def retrieve(self, query: str) -> list[dict]:
        """Retrieve the top_k most relevant chunks for a given query."""
        
        query = "Represent this sentence for searching relevant passages: " + query
        query_embed = self.model.encode([query], normalize_embeddings=True, convert_to_numpy=True)        
        # Perform the search
        ids, distances = self.index.knn_query(query_embed, self.top_k)
        ids, distances = ids[0], distances[0]
        
        # Compile the results
        results = []
        for chunk_id, dist in zip(ids, distances):
            chunk = self.chunks[chunk_id]
            
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
    
    # Load chunks
    with open(CHUNKS_PATH, 'r', encoding= 'utf-8') as f:
        chunks = json.load(f)
        
    Retriver = Retriever(MODEL, chunks, HNSW_PATH, SPACE, EF, DIM, K)
    
    query = input("Enter a query: ")
    results = Retriver.retrieve(query)
    
    for i, result in enumerate(results):
        print(f"\nResult {i + 1}:")
        print(f"Score: {100*result['score']:.4f}%")
        print(f"Distance: {result['distance']:.4f}")
        print(f"Chunk ID: {result['chunk id']}")
        print(f"Article ID: {result['article id']}")
        print(f"Title: {result['title']}")
        print(f"Text: {result['text'][:200]}...")  # Print first 200 characters of the text