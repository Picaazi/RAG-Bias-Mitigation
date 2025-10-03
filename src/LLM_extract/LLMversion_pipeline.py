import os
import time
import pandas as pd
import openai
import sys

from llm_extraction import extract_bias_groups  # Updated LLM extraction function

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from retriever import Retriever
from embedders import Embedder
from decomposition import decompose_query, combine_queries
from rewriting import rewrite_query
from metrics import doc_overlap, sem_similarity, representation_variance


RESULTS_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
os.makedirs(RESULTS_FOLDER, exist_ok=True)

def llm_pipeline(questions, docs_per_question, all_docs, k=5, mode="Decompose"):
    """
    docs_per_question: list of lists of documents per question (for base retrieval)
    all_docs: full combined document set (for final retrieval)
    """
    overlap_scores = []
    sem_scores = []
    rep_variance_scores = []

    final_questions = []
    final_results = []
    base_results = []
    doc_bias_annotations_all = []
    new_synonyms_all = []

    print("Initializing embedder")
    eb = Embedder(use_flagmodel=False)

    # --- Helper to safely get doc text ---
    def get_doc_text(item):
        if isinstance(item, dict):
            return item.get("doc", "")
        elif isinstance(item, (list, tuple)):
            return item[0]  # adjust if text is at index 1
        return str(item)

    for i, q in enumerate(questions):
        d = docs_per_question[i]
        print(f"\n=== Processing question {i+1}/{len(questions)} ===")
        print(f"Question: {q}")

        # Base retrieval from per-question docs
        base_retriever = Retriever(d, embedder=eb)
        base_result = base_retriever.retrieve(q, top_k=min(k, len(d)))
        base_docs = [get_doc_text(doc) for doc in base_result]

        # Final retrieval from combined CSV
        final_retriever = Retriever(all_docs, embedder=eb)

        if mode.lower() == "decompose":
            sub_qs = decompose_query(q)
            combined_qs = combine_queries(sub_qs)
            result = final_retriever.retrieve(query=combined_qs, top_k=min(k, len(all_docs)))
            final_questions.append(sub_qs)
        elif mode.lower() == "rewrite":
            new_q = rewrite_query([q])
            result = final_retriever.retrieve(query=new_q[0], top_k=min(k, len(all_docs)))
            final_questions.append(new_q)
        elif mode.lower() == "both":
            sub_qs = decompose_query(q)
            for j, sub_q in enumerate(sub_qs):
                bias_df, new_synonyms = extract_bias_groups([get_doc_text(sub_q)], save_csv=False)
                if not bias_df.empty and bias_df["subgroup"].iloc[0] != "None":
                    new_q = rewrite_query([get_doc_text(sub_q)])
                    sub_qs[j] = new_q[0]
            combined_qs = combine_queries(sub_qs)
            result = final_retriever.retrieve(query=combined_qs, top_k=min(k, len(all_docs)))
            final_questions.append(sub_qs)

        result_docs = [get_doc_text(doc) for doc in result]
        final_results.append(result_docs)
        base_results.append(base_docs)

        # === Metrics ===
        base_embed = eb.encode_queries(base_docs)

        # LLM extraction (DataFrame + new synonyms)
        doc_bias_df, new_synonyms_dict = extract_bias_groups(result_docs, save_csv=False)
        new_synonyms_all.append(new_synonyms_dict)

        # Build group_set (combine matched subgroups + new synonyms)
        group_set = {}
        for category in doc_bias_df["category"].unique():
            if category == "None":
                continue
            matched_terms = doc_bias_df[doc_bias_df["category"] == category]["subgroup"].unique().tolist()
            extra_terms = new_synonyms_dict.get(category, [])
            group_set[category] = list(set(matched_terms + extra_terms))

        #Categorization
        print("\n--- Bias Group Mapping ---")
        for category, terms in group_set.items():
            print(f"Category: {category}")
            print(f"  Matched subgroups & new terms: {terms}")
        print("--------------------------\n")


        # Per-group overlap & semantic similarity
        group_overlap_scores = []
        group_sem_scores = []

        for group, terms in group_set.items():
            g_docs = [
                d_ for d_, sg in zip(result_docs, doc_bias_df["subgroup"]) if sg == group
            ]
            if not g_docs:
                continue

            group_overlap_scores.append(doc_overlap(base_docs, g_docs))
            g_embeds = eb.encode_queries(g_docs)
            group_sem_scores.append(sem_similarity(base_embed, g_embeds))

        # Fallback if no groups
        if group_overlap_scores:
            overlap = sum(group_overlap_scores) / len(group_overlap_scores)
        else:
            overlap = doc_overlap(base_docs, result_docs)

        if group_sem_scores:
            sem = sum(group_sem_scores) / len(group_sem_scores)
        else:
            sem = sem_similarity(base_embed, eb.encode_queries(result_docs))

        # Representation variance still uses group_set
        rep_var = representation_variance(
            documents=result_docs,
            embedder=eb,
            group_set=group_set,
        )

        # --- Append metrics ---
        overlap_scores.append(overlap)
        sem_scores.append(sem)
        rep_variance_scores.append(rep_var)
        doc_bias_annotations_all.append(doc_bias_df)

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
    print(f"Metrics saved to {metrics_csv}")

    # Save bias extraction CSV
    if doc_bias_annotations_all:
        all_bias_df = pd.concat(doc_bias_annotations_all, ignore_index=True)
        bias_csv = os.path.join(RESULTS_FOLDER, f"doc_bias_{mode}_{timestamp}.csv")
        all_bias_df.to_csv(bias_csv, index=False)
        print(f"Document-level bias annotations saved to {bias_csv}")

    return metrics_df, doc_bias_annotations_all, new_synonyms_all


