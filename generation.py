import sys
import yaml
import json
import requests
from typing import Generator
from pathlib import Path
from Retrieval.Retrieve import Retriever
from ingest import ingest_article

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
PROMPTS = {"Default": LLM["Prompts"]["Default"],
           "Reformat": LLM["Prompts"]["Reformat"]}

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
    def __init__(self, llm_name: str, llm_url: str, prompts: dict[str, str]) -> None:
        
        with open(CHUNKS_PATH, 'r') as f:
            chunks = json.load(f)

        self.llm_name = llm_name
        self.llm_url = llm_url
        self.prompts = prompts
        self.retriever = Retriever(EMBED_MODEL, chunks, HNSW_PATH, SPACE, EF ,DIM, K)
        
    def format(self, query: str, retrieved_chunks: list[dict] | None) -> str:
        """Format the query and retrieved chunks into a prompt for the LLM."""
        if not retrieved_chunks:
            return f"{self.prompts["Reformat"]}\n\nQuestion: {query}\nAnswer:"
        
        context = "\n\n".join([f"Trust accuracy: {chunk['score']}\nTitle: {chunk['title']}\nText: {chunk['text']}" for chunk in retrieved_chunks])
        prompt = f"{self.prompts["Default"]}\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"
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
    
    def add_to_db(self, query: str) -> None:
        """Add a new info to the database."""
        prompt = self.format(query, retrieved_chunks= None)
        response = requests.post(
            self.llm_url,
            json={"model": self.llm_name, "prompt": prompt, "stream": False},
        )
        response.raise_for_status()
        response = response.json()["response"]
        
        title = input("Give a title to the article: ")
            
        # Save the article to the database
        ingest_article(title, response)
        
    
    def _streamed_response(self, response: requests.Response) -> Generator:
        """Handle streaming responses from the LLM API."""
        for line in response.iter_lines():
            if not line:
                continue
            payload = json.loads(line)
            if payload.get("done"):
                break
            yield payload["token"] if "token" in payload else payload.get("response", "")
    
    def handle_query(self, query: str, stream: bool = False) -> str | Generator:
        """Single entry point: route by prefix if given, else classify via LLM."""
        
        if query.startswith("/add "):
            query = query.removeprefix("/add").strip()
            
            self.add_to_db(query)
            return "Added to the knowledge base. Restart the session to update db"
        
        return self.generate(query, stream)
    
if __name__ == "__main__":
    llm_rag = LLM_RAG(LLM_MODEL, LLM_URL, PROMPTS)
    
    print("Use '/add' to add data to db")

    query = input("Enter a query: ")
    response = llm_rag.handle_query(query, stream=True)

    print("LLM: ", end="")
    for chunk in response:
        print(chunk, end="", flush=True)
    print()