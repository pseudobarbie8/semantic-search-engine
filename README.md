# semantic-search-engine

Hybrid search (BM25 + dense embeddings + reranker) over arXiv abstracts, with a small web UI.

```bash
pip install -r requirements.txt
python build_index.py --data data/arxiv_abstracts.jsonl
python eval.py            # compares bm25 / dense / hybrid
streamlit run app.py
```

## Results

| Method | Recall@10 | MRR |
|---|---|---|
| BM25 | - | - |
| Dense | - | - |
| Hybrid | - | - |
| Hybrid + rerank | - | - |

Fill this in after running `eval.py`.

## How it works

- BM25 via `rank_bm25`
- Dense vectors with sentence-transformers, stored in FAISS
- Scores merged with reciprocal rank fusion, top results reranked with a cross-encoder

## Layout

```
build_index.py   indexing
search.py        retrieval + fusion + rerank
eval.py          metrics
app.py           Streamlit UI
```

MIT
