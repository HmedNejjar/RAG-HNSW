import yaml
import json
import numpy as np
import hnswlib
from pathlib import Path
from sentence_transformers import SentenceTransformer

from Datasets.load_ds import add_article
from chunking import chunk_new_article
from Embedding import embed
from hnsw import add_single

ROOT = Path(__file__).parent

with open(ROOT / "config.yaml", 'r') as f:
    config = yaml.safe_load(f)

EMBED = config["Embedding"]
CHUNK = config["Chunking"]
HNSW = config["HNSW"]

EMBED_MODEL = EMBED["Model"]

ARTICLES_PATH = ROOT / "Datasets" / "wiki_articles.json"
EMBED_PATH = ROOT / EMBED["Savepath"]
CHUNKS_PATH = ROOT / CHUNK["Savepath"]
HNSW_PATH = ROOT / HNSW["Savepath"]

BATCH = EMBED["Batch"]
CHUNK_SIZE = CHUNK["Chunk size"]
OVERLAP = CHUNK["Overlap"]

SPACE = HNSW["Distance Metric"]
K = HNSW["top_k"]
EF = HNSW["EF_search"]
NUM_ELEMENTS = HNSW["number of elements"]
DIM = HNSW["dim"]

def ingest_article(title: str, text: str) -> None:
    """Run one new document through the full ingestion pipeline: 
    article -> chunks -> embeddings -> HNSW index.
    """
    
    # 1. Load existing articles and append
    with open(ARTICLES_PATH, 'r', encoding= 'utf-8') as f:
        articles = json.load(f)
        
    articles, new_article = add_article(articles, title, text)
    
    if new_article is None:
        print("Duplicate article detected, skipping.")
        return None
    
    with open(ARTICLES_PATH, 'w', encoding='utf-8') as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)
        
    # 2. Chunking the new article
    with open(CHUNKS_PATH, 'r', encoding= 'utf-8') as f:
        chunks = json.load(f)
        
    new_article_index = len(articles) - 1
    new_chunks = chunk_new_article(new_article, new_article_index, CHUNK_SIZE, OVERLAP, start_chunk_id=len(chunks))
    
    chunks.extend(new_chunks)
    with open(CHUNKS_PATH, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    
    # 3. Embedding the new chunks
    texts = [chunk["text"] for chunk in new_chunks]
    
    embed_model = SentenceTransformer(EMBED_MODEL)
    new_embeddings = embed(texts, embed_model, batch_size= BATCH)
    existing_embeddings = np.load(EMBED_PATH) if EMBED_PATH.exists() else None
    
    final_embeddings = (new_embeddings if existing_embeddings is None else np.vstack([existing_embeddings, new_embeddings]))
    
    np.save(EMBED_PATH, final_embeddings)
    
    # 4. Adding new embeddings to the HNSW index
    index = hnswlib.Index(space=SPACE, dim=DIM)
    
    if HNSW_PATH.exists():
        index.load_index(str(HNSW_PATH))
        index = add_single(index, new_embeddings)
        index.save_index(str(HNSW_PATH))
    
    else:
        index.init_index(max_elements= 2*final_embeddings.shape[0], M= HNSW["Max Edges"], ef_construction= HNSW["EF_construct"])
        index.add_items(final_embeddings, np.arange(final_embeddings.shape[0]))
        index.save_index(str(HNSW_PATH))
    
    current_count = index.get_current_count()
    with open(ROOT / "config.yaml", 'w') as f:
        config["HNSW"]["number of elements"] = current_count
        yaml.safe_dump(config, f)
        
    print(f"Ingested '{title}': {len(new_chunks)} chunks added. Index now holds {index.get_current_count()} vectors.")
    
        