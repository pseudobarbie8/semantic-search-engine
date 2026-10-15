"""BM25, dense, hybrid (reciprocal rank fusion) and cross-encoder reranking."""
import json
import re

import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer


def tok(s):
    return re.findall(r"\w+", s.lower())


class Searcher:
    def __init__(self, index_dir="index"):
        self.docs = json.load(open(f"{index_dir}/docs.json"))
        self.bm25 = BM25Okapi([tok(d["text"]) for d in self.docs])
        self.index = faiss.read_index(f"{index_dir}/dense.faiss")
        self.enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self._reranker = None

    def bm25_search(self, q, k=10):
        scores = self.bm25.get_scores(tok(q))
        return sorted(range(len(scores)), key=lambda i: -scores[i])[:k]

    def dense_search(self, q, k=10):
        v = self.enc.encode([q], normalize_embeddings=True)
        return [i for i in self.index.search(v, k)[1][0].tolist() if i >= 0]

    def hybrid_search(self, q, k=10, c=60):
        """Reciprocal rank fusion of the BM25 and dense rankings."""
        fused = {}
        for ranking in (self.bm25_search(q, 50), self.dense_search(q, 50)):
            for rank, i in enumerate(ranking):
                fused[i] = fused.get(i, 0.0) + 1.0 / (c + rank + 1)
        return sorted(fused, key=fused.get, reverse=True)[:k]

    def rerank(self, q, ids, k=10):
        if self._reranker is None:
            self._reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        scores = self._reranker.predict([(q, self.docs[i]["text"]) for i in ids])
        return [i for _, i in sorted(zip(scores, ids), key=lambda x: -x[0])][:k]
