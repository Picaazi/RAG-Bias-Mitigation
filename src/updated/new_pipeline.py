import sys
import os
import time
import numpy as np
import pandas as pd

# -----------------------------
# Add project root to Python path
# -----------------------------
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, SRC_DIR)

# -----------------------------
# Import custom modules
# -----------------------------
from doc_loaders import build_genderbias_eval_index
from retriever import Retriever
from embedders import Embedder
from decomposition import decompose_query, combine_queries
from rewriting import rewrite_query

# ==============================
# Semantic Metrics
# ==============================

def semantic_recall_at_k(retrieved_docs, base_docs, embedder, k=5, threshold = 0.8):
    """
    Fraction of base docs that have at least one retrieved doc (top-k) 
    semantically similar (highest cosine similarity) regardless of threshold.
    """
    retrieved_emb = embedder.encode_queries(retrieved_docs[:k])
    base_emb = embedder.encode_queries(base_docs)
    
    hits = 0
    for b in base_emb:
        # cosine similarities with top-k retrieved docs
        sims = (retrieved_emb @ b) / (np.linalg.norm(retrieved_emb, axis=1) * np.linalg.norm(b) + 1e-8)
        hits += 1 if np.max(sims) > 0 else 0  # count as hit if any similarity > 0
    return hits / max(len(base_docs), 1)


def semantic_mrr(retrieved_docs, base_docs, embedder, threshold=0.8):
    retrieved_emb = embedder.encode_queries(retrieved_docs)
    base_emb = embedder.encode_queries(base_docs)
    
    for rank, r in enumerate(retrieved_emb, start=1):
        sims = [(r @ b) / (np.linalg.norm(r) * np.linalg.norm(b) + 1e-8) for b in base_emb]
        if max(sims) >= threshold:
            return 1.0 / rank
    return 0.0

def soft_overlap(base_docs, retrieved_docs, embedder):
    """
    Embedding-based overlap score (semantic, robust to paraphrases).
    Returns a value between 0 and 1.
    """
    if not base_docs or not retrieved_docs:
        return 0.0

    base_emb = embedder.encode_queries(base_docs)
    ret_emb = embedder.encode_queries(retrieved_docs)

    scores = []
    for b in base_emb:
        sims = (ret_emb @ b) / (np.linalg.norm(ret_emb, axis=1) * np.linalg.norm(b) + 1e-8)
        scores.append(min(max(np.max(sims), 0.0), 1.0))  # clamp to [0,1]

    return float(np.mean(scores))


# ==============================
# Evaluation Pipeline
# ==============================

RESULTS_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
os.makedirs(RESULTS_FOLDER, exist_ok=True)

def llm_pipeline_eval(eval_index, k=5, mode="rewrite", sem_threshold=0.8):
    """
    Runs retrieval evaluation for each question in eval_index with semantic metrics.
    Saves original and modified questions to CSV.
    """
    embedder = Embedder(use_flagmodel=False)

    # Track results
    questions = []
    modified_questions = []
    all_retrieved_docs = []
    all_base_docs = []
    recall_scores = []
    mrr_scores = []
    soft_overlap_scores = []

    for qid, item in eval_index.items():
        q = item["question"]
        eval_docs = item["docs"]
        base_docs = item["base_docs"]

        # -----------------------------
        # Query modification
        # -----------------------------
        if mode.lower() == "decompose":
            sub_qs = decompose_query(q)
            query = combine_queries(sub_qs)
        elif mode.lower() == "rewrite":
            rewritten = rewrite_query([q])
            query = rewritten[0] if rewritten else q
        elif mode.lower() == "both":
            sub_qs = decompose_query(q)
            query = combine_queries(sub_qs)
        else:
            query = q

        # -----------------------------
        # Print original and modified question
        # -----------------------------
        print(f"\n=== Question {qid+1}/{len(eval_index)} ===")
        print("Original question:", q)
        print("Modified question:", query)

        # -----------------------------
        # Retrieve top-k documents
        # -----------------------------
        retriever = Retriever(eval_docs, embedder=embedder)
        retrieved = retriever.retrieve(query=query, top_k=min(k, len(eval_docs)))
        retrieved_docs = [str(d["doc"]) if isinstance(d, dict) else str(d) for d in retrieved]

        # -----------------------------
        # Compute semantic metrics
        # -----------------------------
        recall_scores.append(semantic_recall_at_k(retrieved_docs, base_docs, embedder, k=k, threshold=sem_threshold))
        mrr_scores.append(semantic_mrr(retrieved_docs, base_docs, embedder, threshold=sem_threshold))
        soft_overlap_scores.append(soft_overlap(base_docs, retrieved_docs, embedder))

        # -----------------------------
        # Store for CSV
        # -----------------------------
        questions.append(q)
        modified_questions.append(query)
        all_retrieved_docs.append(retrieved_docs)
        all_base_docs.append(base_docs)

    # -----------------------------
    # Save results
    # -----------------------------
    timestamp = int(time.time())
    out_path = os.path.join(RESULTS_FOLDER, f"eval_results_{mode}_{timestamp}.csv")

    df = pd.DataFrame({
        "question": questions,
        "modified_question": modified_questions,
        "base_docs": all_base_docs,
        "retrieved_docs": all_retrieved_docs,
        "recall@k_sem": recall_scores,
        "mrr_sem": mrr_scores,
        "soft_overlap": soft_overlap_scores,
    })

    df.to_csv(out_path, index=False)
    print(f"\n✅ Semantic evaluation results saved to: {out_path}")

    return df
