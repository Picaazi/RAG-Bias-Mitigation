import os
import pickle
import numpy as np
from typing import List, Sequence
from FlagEmbedding import FlagModel
from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self, model_name="BAAI/bge-m3", use_flagmodel=True, cache_path=None, device="cpu"):
        self.model_name = model_name
        self.use_flagmodel = use_flagmodel
        self.cache_path = cache_path
        self.device = device

        if use_flagmodel:
            self.model = FlagModel(model_name, query_instruction_for_retrieval="Represent this query for retrieval:", use_fp16=False)
            self.encode_corpus = self._encode_corpus_flag
            self.encode_queries = self._encode_queries_flag
            self.dim = None
        else:
            self.model = SentenceTransformer(model_name, device=device)
            self.encode_corpus = self._encode_corpus_st
            self.encode_queries = self._encode_queries_st
            self.dim = self.model.get_sentence_embedding_dimension()

    def _encode_corpus_flag(self, texts: Sequence[str], batch_size: int = 64):
        arr = self.model.encode_corpus(list(texts))
        self.dim = arr[0].shape[0] if len(arr) > 0 else self.dim
        return np.asarray(arr)

    def _encode_queries_flag(self, texts: Sequence[str]):
        arr = self.model.encode_queries(list(texts))
        self.dim = arr[0].shape[0] if len(arr) > 0 else self.dim
        return np.asarray(arr)

    def _encode_corpus_st(self, texts: Sequence[str], batch_size: int = 64):
        emb = self.model.encode(list(texts), batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False)
        self.dim = emb.shape[1]
        return emb

    def _encode_queries_st(self, texts: Sequence[str]):
        emb = self.model.encode(list(texts), convert_to_numpy=True)
        self.dim = emb.shape[1]
        return emb

    def load_or_build_embeddings(self, texts: List[str], cache_path: str = None, force_recompute: bool = False):
        cache_path = cache_path or self.cache_path
        if cache_path and os.path.exists(cache_path) and not force_recompute:
            with open(cache_path, "rb") as f:
                emb = pickle.load(f)
            return np.asarray(emb)

        emb = self.encode_corpus(texts)
        if cache_path:
            with open(cache_path, "wb") as f:
                pickle.dump(emb, f)
        return np.asarray(emb)

    def info(self):
        return {"model_name": self.model_name, "use_flagmodel": self.use_flagmodel, "dim": self.dim}
