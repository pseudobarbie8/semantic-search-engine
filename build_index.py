"""Embed documents and save a FAISS index next to the raw docs."""
import argparse
import json
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/sample.jsonl", help='jsonl with {"id", "title", "text"} per line')
ap.add_argument("--out", default="index")
a = ap.parse_args()

docs = [json.loads(line) for line in open(a.data) if line.strip()]
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
emb = model.encode([d["text"] for d in docs], normalize_embeddings=True, show_progress_bar=True)

index = faiss.IndexFlatIP(emb.shape[1])
index.add(emb)

Path(a.out).mkdir(exist_ok=True)
faiss.write_index(index, f"{a.out}/dense.faiss")
json.dump(docs, open(f"{a.out}/docs.json", "w"))
print(f"indexed {len(docs)} docs -> {a.out}/")
