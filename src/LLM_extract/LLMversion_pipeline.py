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
    if mode not in ["decompose", "rewrite", "both"]:
        raise ValueError("Invalid mode. Choose from 'decompose', 'rewrite', or 'both'.")

    overlap_scores = []
    sem_scores = []
    rep_variance_scores = []
    
    final_questions = []
    final_results = []
    base_results = []
    doc_bias_annotations_all = []

    print("Initializing embedder")
    eb = Embedder(use_flagmodel=False)

    for i in range(len(questions)):
        q = questions[i]
        d = docs[i]
        print(f"Processing question {i+1}/{len(questions)}: {q}")
        retriever = Retriever(d, embedder=eb)
        print("Getting base result")
        base_result = retriever.retrieve(q, top_k=min(k, len(d)))
        
        result = []

        if mode == "decompose":
            print("Decomposed sub-queries:")
            sub_qs = decompose_query(q)
            for j, sub_q in enumerate(sub_qs):
                print(f"Sub-query {j+1}: {sub_q}")
            combined_qs = combine_queries(sub_qs)
            result = retriever.retrieve(query=combined_qs, top_k=min(k, len(d)))
            final_questions.append(sub_qs)

        elif mode == "rewrite":
            print("Rewriting sub-queries:")
            new_q = rewrite_query([q])
            print(f"Original query: {q}")
            print(f"Rephrased query: {new_q[0]}")
            result = retriever.retrieve(query=new_q[0], top_k=min(k, len(d)))
            final_questions.append(new_q)

        elif mode == "both":
            print("Decomposing and rewriting sub-queries:")
            sub_qs = decompose_query(q)
            
            # LLM-based bias detection
            sub_qs_biased = []
            for sub_q in sub_qs:
                bias_info = extract_bias_groups([sub_q], save_csv=False)
                is_biased = not bias_info.empty and bias_info["subgroup"].iloc[0] != "None"
                sub_qs_biased.append(is_biased)

            for j, sub_q in enumerate(sub_qs):
                if sub_qs_biased[j]:
                    new_q = rewrite_query(sub_q)
                    print(f"Sub-query {j+1} is biased: {sub_q}")
                    print(f"Rephrased to neutral: {new_q}")
                    sub_qs[j] = new_q[0]
                else:
                    print(f"Sub-query {j+1} is neutral: {sub_q}")

            combined_qs = combine_queries(sub_qs)
            result = retriever.retrieve(query=combined_qs, top_k=min(k, len(d)))
            final_questions.append(sub_qs)

        final_docs = [d["doc"] for d in result]
        base_docs = [d["doc"] for d in base_result]
        final_results.append(final_docs)
        base_results.append(base_docs)

        base_embed = eb.encode_queries(base_docs)
        result_embed = eb.encode_queries(final_docs)

        # Metrics
        overlap_scores.append(doc_overlap(base_docs, final_docs))
        sem_scores.append(sem_similarity(base_embed, result_embed))
        rep_variance_scores.append(representation_variance(final_docs, embedder=eb))

        # LLM extraction for retrieved documents
        doc_bias_df = extract_bias_groups(final_docs, save_csv=False)
        doc_bias_df["question"] = q
        doc_bias_annotations_all.append(doc_bias_df)

    # Save metrics
    metrics_df = pd.DataFrame({
        "question": questions,
        "base_result": base_results,
        "final_result": final_results,
        "overlap_score": overlap_scores,
        "sem_score": sem_scores,
        "rep_variance_score": rep_variance_scores
    })

    timestamp = int(time.time())
    metrics_csv = os.path.join(RESULTS_FOLDER, f"results_{mode}_{timestamp}.csv")
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"✅ Metrics saved to {metrics_csv}")

    # Save document-level bias annotations
    if doc_bias_annotations_all:
        all_bias_df = pd.concat(doc_bias_annotations_all, ignore_index=True)
        bias_csv = os.path.join(RESULTS_FOLDER, f"doc_bias_{mode}_{timestamp}.csv")
        all_bias_df.to_csv(bias_csv, index=False)
        print(f"✅ Document-level bias annotations saved to {bias_csv}")


def data_router(name):
    if name not in ["gender_bias", "politics_bias", "bbq", "bibleqa", "islamqa"]:
        raise ValueError(f"Unknown dataset: {name}")

    if name == "gender_bias":
        train, test = dataloader.load_gender_bias()
        data = pd.concat([train, test], ignore_index=True)
        questions = data["question"].tolist()
        docs = []
        for i in range(len(data)):
            d = data.iloc[i]
            docs.append([d["bias1-document1"], d["bias1-document2"], d["bias2-document1"], d["bias2-document2"]])
        return questions, docs
    elif name == "politics_bias":
        train, test = dataloader.load_politics_bias()
        data = pd.concat([train, test], ignore_index=True)
        questions = data["question"].tolist()
        docs = []
        for i in range(len(data)):
            d = data.iloc[i]
            docs.append([d["left_claim"], d["right_claim"]])
        return questions, docs
    elif name == "bbq":
        data = pd.concat(dataloader.load_bbq_datasets(), ignore_index=True)
        # TODO: set questions, docs properly
        return questions, docs
    elif name == "bibleqa":
        data = dataloader.load_bibleqa()
        questions = data["Question"].tolist()
        docs = []
        for i in range(len(data)):
            d = data.iloc[i]
            docs.append([d["KJV_Verse"], d["ASV_Verse"], d["YLT_Verse"], d["WEB_Verse"]])
        return questions, docs
    elif name == "islamqa":
        return dataloader.load_islamqa()


if __name__ == "__main__":
    print("Loading datasets...")
    questions, docs = data_router("gender_bias")
    q = questions[:5]
    d = docs[:5]
    llm_pipeline(q, d, mode="both")