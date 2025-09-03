import os
import time
import pandas as pd
import openai
from retriever import Retriever
from embedders import Embedder
from decomposition import decompose_query, combine_queries
from rewriting import rewrite_query
from metrics import doc_overlap, sem_similarity, representation_variance
import multi_dataset_loader as dataloader
from llm_extraction import extract_bias_groups  # Your LLM CSV extraction function

RESULTS_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
os.makedirs(RESULTS_FOLDER, exist_ok=True)

def llm_pipeline(questions, docs, k=5, mode="Decompose"):
    overlap_scores = []
    sem_scores = []
    rep_variance_scores = []

    final_questions = []
    final_results = []
    base_results = []
    doc_bias_annotations_all = []

    print("Initializing embedder")
    eb = Embedder(use_flagmodel=False)

    for i, q in enumerate(questions):
        d = docs[i]
        print(f"\nProcessing question {i+1}/{len(questions)}: {q}")
        retriever = Retriever(d, embedder=eb)
        
        # Base retrieval
        base_result = retriever.retrieve(q, top_k=min(k, len(d)))
        base_docs = [doc["doc"] for doc in base_result]

        # === Query handling ===
        if mode.lower() == "decompose":
            sub_qs = decompose_query(q)
            combined_qs = combine_queries(sub_qs)
            result = retriever.retrieve(query=combined_qs, top_k=min(k, len(d)))
            final_questions.append(sub_qs)
        elif mode.lower() == "rewrite":
            new_q = rewrite_query([q])
            result = retriever.retrieve(query=new_q[0], top_k=min(k, len(d)))
            final_questions.append(new_q)
        elif mode.lower() == "both":
            sub_qs = decompose_query(q)
            for j, sub_q in enumerate(sub_qs):
                bias_info = extract_bias_groups([sub_q], save_csv=False)
                if not bias_info.empty and bias_info["subgroup"].iloc[0] != "None":
                    new_q = rewrite_query(sub_q)
                    sub_qs[j] = new_q[0]
            combined_qs = combine_queries(sub_qs)
            result = retriever.retrieve(query=combined_qs, top_k=min(k, len(d)))
            final_questions.append(sub_qs)

        result_docs = [doc["doc"] for doc in result]
        final_results.append(result_docs)
        base_results.append(base_docs)

        # === Metrics ===
        base_embed = eb.encode_queries(base_docs)
        result_embed = eb.encode_queries(result_docs)

        overlap_scores.append(doc_overlap(base_docs, result_docs))
        sem_scores.append(sem_similarity(base_embed, result_embed))

        # LLM extraction for representation variance
        doc_bias_df = extract_bias_groups(result_docs, save_csv=False)
        doc_bias_annotations_all.append(doc_bias_df)

        rep_var = representation_variance(
            documents=result_docs,
            embedder=eb,
            group_set={s: [s] for s in doc_bias_df["subgroup"].unique() if s != "None"},
        )
        rep_variance_scores.append(rep_var)

    # === Save metrics ===
    timestamp = int(time.time())
    metrics_df = pd.DataFrame({
        "question": questions,
        "base_result": base_results,
        "final_result": final_results,
        "overlap_score": overlap_scores,
        "sem_score": sem_scores,
        "rep_variance_score": rep_variance_scores
    })
    metrics_csv = os.path.join(RESULTS_FOLDER, f"results_{mode}_{timestamp}.csv")
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"✅ Metrics saved to {metrics_csv}")

    # Save bias extraction CSV
    if doc_bias_annotations_all:
        all_bias_df = pd.concat(doc_bias_annotations_all, ignore_index=True)
        bias_csv = os.path.join(RESULTS_FOLDER, f"doc_bias_{mode}_{timestamp}.csv")
        all_bias_df.to_csv(bias_csv, index=False)
        print(f"✅ Document-level bias annotations saved to {bias_csv}")
