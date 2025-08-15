import data_reading
from decomposition import decompose_query, combine_queries
from rewriting import rewrite_query
from bias_detection import detect_bias
from metrics import doc_overlap, sem_similarity, representation_variance
import bm25s
import pandas as pd
from client import get_openai_embedding
from retriever import Retriever
from embedders import Embedder

def pipeline(questions, docs, k=5, mode="Decompose"):
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
        base_result = retriever.retrieve(bm25s.tokenize(q), k=k)
        
        result = []
        if mode == "decompose":
            print("Decomposed sub-queries:")
            sub_qs = decompose_query(q)
            for j, sub_q in enumerate(sub_qs):
                print(f"Sub-query {j+1}: {sub_q}")
            combined_qs = combine_queries(sub_qs)
            result = retriever.retrieve(query=combined_qs, top_k=k)
            final_questions.append(sub_qs)
        elif mode == "rewrite":
            print("Rewriting sub-queries:")
            new_q = rewrite_query([q])
            print(f"Original query: {q}")
            print(f"Rephrased query: {new_q[0]}")
            result = retriever.retrieve(query=new_q, top_k=k)
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
            result = retriever.retrieve(query=combined_qs, top_k=k)
            final_questions.append(sub_qs)
        
        final_results.append(result)
        base_results.append(base_result)

        base_embed = eb.encode_queries(base_result)
        result_embed = eb.encode_queries(result)
        # Calculate metrics
        overlap_scores.append(doc_overlap(base_result, result))
        sem_scores.append(sem_similarity(base_embed, result_embed))
        # rep_variance_scores.append(representation_variance(results, group_set=group_set))

    # Save the results as csv
    results_df = pd.DataFrame({
        "question": questions,
        "base_result": base_results,
        "final_result": final_results,
        "overlap_score": overlap_scores,
        "sem_score": sem_scores,
        "rep_variance_score": rep_variance_scores
    })
    results_df.to_csv(f"results/results_{mode}.csv", index=False)

if __name__ == "__main__":
    
    # Step 1: Dataset loading (one-by-one /all) - > queries answers and corpus
    print("Loading datasets...")
    train, test = data_reading.read_gender_bias_data()
    print("Data loaded successfully.")
    print(f"Number of training samples: {len(train)}")
    print(f"Number of testing samples: {len(test)}")

    # Put training data set into a list of questions and a list of documents
    train_questions = train["question"].tolist()
    train_docs = []
    for i in range(len(train)):
        d = train.iloc[i]
        docs= [d["bias1-document1"], d["bias1-document2"], d["bias2-document1"], d["bias2-document2"]]
        train_docs.append(docs)
    pipeline(train_questions, train_docs, mode="rewrite")
    # decompose method is not finished, not sure how to retrieve document by many sub-queries
    # same for method both
    # rewrite method is debugging