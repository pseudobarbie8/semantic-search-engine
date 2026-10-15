"""Recall@10 and MRR for each retrieval method."""
import argparse
import json

from search import Searcher

ap = argparse.ArgumentParser()
ap.add_argument("--queries", default="data/sample_queries.jsonl", help='jsonl with {"query", "relevant": doc_id}')
a = ap.parse_args()

s = Searcher()
queries = [json.loads(line) for line in open(a.queries) if line.strip()]
methods = {
    "bm25": s.bm25_search,
    "dense": s.dense_search,
    "hybrid": s.hybrid_search,
    "hybrid+rerank": lambda q, k=10: s.rerank(q, s.hybrid_search(q, 50), k),
}

print("| Method | Recall@10 | MRR |\n|---|---|---|")
for name, fn in methods.items():
    hits, rr = 0, 0.0
    for item in queries:
        ids = [s.docs[i]["id"] for i in fn(item["query"], 10)]
        if item["relevant"] in ids:
            hits += 1
            rr += 1.0 / (ids.index(item["relevant"]) + 1)
    print(f"| {name} | {hits / len(queries):.3f} | {rr / len(queries):.3f} |")
