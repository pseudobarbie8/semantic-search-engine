# semantic-search-engine

Hybrid search (BM25 + dense embeddings + reranker) over a document collection, with a small web UI.

```bash
pip install -r requirements.txt
python build_index.py --data data/sample.jsonl
python eval.py --queries data/sample_queries.jsonl    # compares bm25 / dense / hybrid / rerank
streamlit run app.py
```

## Data formats

```json
{"id": "d1", "title": "...", "text": "..."}
{"query": "...", "relevant": "d1"}
```

`data/sample.jsonl` is 8 toy documents, enough to check the pipeline runs. For real numbers, index a real corpus (arXiv abstracts, a Wikipedia subset) and write your own labelled queries.

## Results

| Method | Recall@10 | MRR |
|---|---|---|
| BM25 | - | - |
| Dense | - | - |
| Hybrid | - | - |
| Hybrid + rerank | - | - |

Fill this in from `eval.py` on a real corpus.

## How it works

- BM25 via `rank_bm25`
- Dense vectors with sentence-transformers (`all-MiniLM-L6-v2`), stored in FAISS
- BM25 and dense rankings merged with reciprocal rank fusion, then reranked with a cross-encoder

## Layout

```
build_index.py   embed docs, write FAISS index
search.py        bm25, dense, hybrid, rerank
eval.py          Recall@10 and MRR
app.py           Streamlit UI
```

MIT
