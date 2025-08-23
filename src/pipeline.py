import data_reading
from decomposition import decompose_query, combine_queries
from rewriting import rewrite_query
from bias_detection import detect_bias
from metrics import doc_overlap, sem_similarity, representation_variance
import bm25s
import pandas as pd
import multi_dataset_loader as dataloader
import corpus_load_read
from retriever import Retriever
from embedders import Embedder
import time
import corpus_load_read
import os

RESULTS_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")

def pipeline(questions, docs, k=5, mode="Decompose", result_folder=RESULTS_FOLDER):
    '''
    Processes a set of questions and documents using different query modification strategies
    and evaluates the retrieval performance.
    This function implements a document retrieval pipeline that can decompose queries into
    sub-queries, rewrite queries to reduce bias, or combine both approaches. It compares
    the results against baseline retrieval and computes various similarity metrics.
    Args:
        questions (list): List of input questions/queries to process
        docs (list): List of document collections corresponding to each question
        k (int, optional): Number of top documents to retrieve. Defaults to 5.
        mode (str, optional): Processing mode - "decompose" for query decomposition,
                             "rewrite" for query rewriting, "both" for decomposition
                             with bias detection and rewriting. Defaults to "Decompose".
    Returns:
        None: Results are saved to CSV file in results/ directory
    Raises:
        ValueError: If mode is not one of "decompose", "rewrite", or "both"
    Output Files:
        Creates a CSV file "results/results_{mode}.csv" containing:
        - Original questions
        - Base retrieval results
        - Final processed results
        - Document overlap scores
        - Semantic similarity scores
        - Representation variance scores (placeholder)
    Note:
        The function prints progress information during execution and requires
        various utility functions (decompose_query, rewrite_query, detect_bias)
        and classes (Embedder, Retriever) to be available in scope.
    '''
    
    if mode not in ["decompose", "rewrite", "both"]:
        raise ValueError("Invalid mode. Choose from 'decompose', 'rewrite', or 'both'.")

    # embed mode?
    overlap_scores = []
    sem_scores = []
    rep_variance_scores = []
    
    final_questions = []
    final_results = []
    base_results = []

    print("Initializing embedder")
    eb = Embedder(use_flagmodel=False)

    # For each data, get question and documents, put them into retriever
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
            sub_qs_biased = detect_bias(sub_qs)
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
        # Calculate metrics
        overlap_scores.append(doc_overlap(base_docs, final_docs))
        sem_scores.append(sem_similarity(base_embed, result_embed)) # Why are we calling this as Semantic Similarity? Higher of this means more divergence???
        rep_variance_scores.append(representation_variance(final_docs, embedder=eb))
        # TODO: add correctness score

    # Save the results as csv
    results_df = pd.DataFrame({
        "question": questions,
        "base_result": base_results,
        "final_result": final_results,
        "overlap_score": overlap_scores,
        "sem_score": sem_scores,
        "rep_variance_score": rep_variance_scores
    })
    
    timestamp = int(time.time())
    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    save_path = os.path.join(RESULTS_FOLDER, f"results_{mode}_{timestamp}.csv")
    results_df.to_csv(save_path, index=False)
    print(f"Results saved to {save_path}")


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
        # Get question and docs
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
        # combine all bbq dataset, return question and docs
        
    ## TODO: load all dataset combined
        return dataloader.load_islamqa()



def corpus_router(name):
    """
    Routes corpus loading based on the corpus name.
    
    Args:
        name (str): Name of the corpus to load
        
    Returns:
        The loaded corpus data
        
    Raises:
        ValueError: If corpus name is not supported
    """
    if name == "wiki":
        data = corpus_load_read.Wikipedia()
    elif name == "polnli":
        data = corpus_load_read.PolNLI()
    elif name == "fever":
        data = corpus_load_read.FEVER()
    elif name == "msmarco":
        data = corpus_load_read.MSMarcoDataset()
    elif name == "sbic":
        data = corpus_load_read.SBIC()
    elif name == "bbc":
        data = corpus_load_read.BBC()
    elif name == "nq":
        data = corpus_load_read.NaturalQuestions()
    elif name == "c4corpus":
        data = corpus_load_read.C4Corpus()
    data.process()
    return data.read()

if __name__ == "__main__":
    
    print("Loading datasets...")
    questions, docs = data_router("gender_bias")
    q = questions[:5]
    d = docs[:5]
    
    pipeline(q, d, mode="both")
