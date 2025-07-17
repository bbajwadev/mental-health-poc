# src/retrieval.py
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer, util

# Load data
DATA_PATH = Path(__file__).parent.parent / "data/processed/conversations.json"
records = json.loads(DATA_PATH.read_text())
contexts  = [r["Context"]  for r in records]
responses = [r["Response"] for r in records]

# Create embeddings once
model = SentenceTransformer("all-MiniLM-L6-v2")
ctx_embeddings = model.encode(contexts, convert_to_tensor=True)

def semantic_fetch(query: str, k: int = 3):
    # Encode query
    q_emb = model.encode(query, convert_to_tensor=True)
    # Compute cosine similarities
    sims = util.cos_sim(q_emb, ctx_embeddings)[0]
    top = sims.topk(k)
    return [(contexts[int(i)], responses[int(i)]) for i in top.indices]
