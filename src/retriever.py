import numpy as np
from rank_bm25 import BM25Okapi
from typing import List, Dict
from src.embedders import Embedder

class Retriever:
    def __init__(self, docs: List[str], method="dense", embedder=None, embeddings_cache=None):
        self.docs = docs
        self.method = method
        self.embedder = embedder

        if method == "dense":
            self.embeddings = embedder.load_or_build_embeddings(docs, cache_path=embeddings_cache)
        elif method == "bm25":
            tokenized_corpus = [doc.split() for doc in docs]
            self.bm25 = BM25Okapi(tokenized_corpus)
        else:
            raise ValueError(f"Unknown retrieval method: {method}")

    def retrieve(self, query: str, top_k=5) -> Dict:
        if self.method == "dense":
            q_emb = self.embedder.encode_queries([query])[0]
            scores = np.dot(self.embeddings, q_emb) / (np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(q_emb))
        else:
            scores = self.bm25.get_scores(query.split())

        top_idx = np.argsort(scores)[::-1][:top_k]
        results = [
            {"rank": i + 1, "score": float(scores[idx]), "doc": self.docs[idx], "doc_id": idx}
            for i, idx in enumerate(top_idx)
        ]
        return {"query": query, "results": results}
