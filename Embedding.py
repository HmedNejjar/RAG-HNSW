import yaml
import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).parent

with open(ROOT / "config.yaml", 'r') as f:
    config = yaml.safe_load(f)
    
EMBED_CONFIG = config["Embedding"]
MODEL = EMBED_CONFIG["Model"]
BATCH = EMBED_CONFIG["Batch"]
SAVE_EMBED = ROOT / EMBED_CONFIG["Savepath"]

def load_model(model_name: str) -> SentenceTransformer:
    """Load a SentenceTransformer model by name."""
    return SentenceTransformer(model_name)

def embed(text: list[str], model_name: str, batch_size: int) -> np.ndarray:
    """Generate embeddings for the given text using the specified model."""
    model = load_model(model_name)
    embeddings = model.encode(text, batch_size= batch_size, show_progress_bar= True, normalize_embeddings= True, convert_to_numpy= True)
    
    print(f"Embedding matrix shape: {embeddings.shape}")
    
    return embeddings
    
if __name__ == "__main__":
    with open(ROOT / "Datasets\\chunks.json", 'r', encoding= "utf-8") as f:
        chunks = json.load(f)
    print(f"Loaded {len(chunks)} chunks.")
    
    # Figure out how many are already embedded
    if SAVE_EMBED.exists():
        existing = np.load(SAVE_EMBED)
        num_existing = existing.shape[0]
    else:
        existing = None
        num_existing = 0
        
    new_chunks = chunks[num_existing:]
    print(f"{num_existing} chunks already embedded. {len(new_chunks)} new chunks to embed.")
    
    if not new_chunks:
        print("Nothing new to embed.")
    
    else:
        texts = [c["text"] for c in chunks]
    
        embeddings = embed(texts, MODEL, BATCH)
    
        # Save embeddings to a file
        np.save(SAVE_EMBED, embeddings)
        print(f"Saved to {SAVE_EMBED}")