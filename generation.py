import sys
import yaml
import json
import requests
from typing import Generator
from pathlib import Path
from Retrieval.Retrieve import Retriever

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))


with open(ROOT / "config.yaml") as f:
    config = yaml.safe_load(f)
    
EMBED = config["Embedding"]
CHUNK = config["Chunking"]
HNSW = config["HNSW"]
LLM = config["LLM"]

LLM_MODEL = LLM["Model"]
LLM_URL = LLM["URL"]
SYS_PROMPT = LLM["System Prompt"]
EMBED_MODEL = EMBED["Model"]

EMBED_PATH = ROOT / EMBED["Savepath"]
CHUNKS_PATH = ROOT / CHUNK["Savepath"]
HNSW_PATH = ROOT / HNSW["Savepath"]

SPACE = HNSW["Distance Metric"]
K = HNSW["top_k"]
EF = HNSW["EF_search"]
NUM_ELEMENTS = HNSW["number of elements"]
DIM = HNSW["dim"]

class LLM_RAG:
    def __init__(self, llm_name: str, llm_url: str) -> None:
        
        with open(CHUNKS_PATH, 'r') as f:
            chunks = json.load(f)

        self.llm_name = llm_name
        self.llm_url = llm_url
        self.retriever = Retriever(EMBED_MODEL, chunks, HNSW_PATH, SPACE, EF ,DIM, K)
        
    def format(self, query: str, retrieved_chunks: list[dict]) -> str:
        """Format the query and retrieved chunks into a prompt for the LLM."""
        context = "\n\n".join([f"Trust accuracy: {chunk['score']}\nTitle: {chunk['title']}\nText: {chunk['text']}" for chunk in retrieved_chunks])
        prompt = f"{SYS_PROMPT}\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"
        return prompt
    
    def generate(self, query: str, stream: bool = False) -> str | Generator:
        """Generate a response from the LLM based on the query and retrieved chunks."""

        # Retrieve relevent chunks from DB
        retrieved_chunks = self.retriever.retrieve(query)

        # Format the prompt for the LLM
        prompt = self.format(query, retrieved_chunks)

        # Send the prompt to the LLM API
        response = requests.post(
            self.llm_url,
            json={"model": self.llm_name, "prompt": prompt, "stream": stream},
            stream=stream,
        )
        response.raise_for_status()

        if stream:
            return self._streamed_response(response)
        return response.json()["response"]
    
    def _streamed_response(self, response: requests.Response) -> Generator:
        """Handle streaming responses from the LLM API."""
        for line in response.iter_lines():
            if not line:
                continue
            payload = json.loads(line)
            if payload.get("done"):
                break
            yield payload["token"] if "token" in payload else payload.get("response", "")
            
    
if __name__ == "__main__":
    llm_rag = LLM_RAG(LLM_MODEL, LLM_URL)

    query = input("Enter a query: ")
    response = llm_rag.generate(query, stream=True)

    print("LLM: ", end="")
    for chunk in response:
        print(chunk, end="", flush=True)
    print()