import data_reading
from decomposition import decompose_query
from rewriting import rewrite_query
from bias_detection import detect_bias
from metrics import doc_overlap, sem_similarity, representation_variance
import bm25s

def get_retrieval_results(question, docs, k=4, mode="None"):

    # Use BM25 as an example for retrieval
    retriever = bm25s.BM25(corpus=docs)
    retriever.index(bm25s.tokenize(docs))
    
    ### Only decomposition and only rewriting
    if mode == "decompose":
        print("Decomposed sub-queries:")
        sub_qs = decompose_query(question)
        for j, sub_q in enumerate(sub_qs):
            print(f"Sub-query {j+1}: {sub_q}")
        results, scores = retriever.retrieve(bm25s.tokenize(sub_qs), k=k)  
    elif mode == "rewrite":
        print("Rewriting sub-queries:")
        new_q = rewrite_query(question)
        print(f"Original query: {question}")
        print(f"Rephrased query: {new_q[0]}")
        results, scores = retriever.retrieve(bm25s.tokenize(new_q), k=k)
    elif mode == "both":
        print("Decomposing and rewriting sub-queries:")
        sub_qs = decompose_query(question)
        sub_qs_biased = detect_bias(sub_qs)
        for j, sub_q in enumerate(sub_qs):
            if sub_qs_biased[j]:
                new_q = rewrite_query(sub_q)
                print(f"Sub-query {j+1} is biased: {sub_q}")
                print(f"Rephrased to neutral: {new_q}")
                sub_qs[j] = new_q[0]
            else:
                print(f"Sub-query {j+1} is neutral: {sub_q}")
        results, scores = retriever.retrieve(bm25s.tokenize(sub_qs), k=k)
    elif mode == "None":
        print("No modification will be made.")
        results, scores = retriever.retrieve(bm25s.tokenize(q), k=k)
    else:
        raise ValueError("Invalid mode. Choose from 'decompose', 'rewrite', 'both', or 'None'.")

    for result, score in zip(results, scores):
            print(f"    {result} (Score: {score})")           
    return results, scores


if __name__ == "__main__":
    """
    Main pipeline function to run the entire process.
    """
    # Step 1: Dataset loading (one-by-one /all) - > queries answers and corpus
    print("Loading datasets...")
    train, test = data_reading.read_gender_bias_data()
    print("Data loaded successfully.")
    print(f"Number of training samples: {len(train)}")
    print(f"Number of testing samples: {len(test)}")
    
    overlap_scores = []
    sem_scores = []
    rep_variance_scores = []
    # For each data, get question and documents, put them into retriever
    for i in range(len(train)):
        d = train.iloc[i]
        q = d["question"]
        docs= [d["bias1-document1"], d["bias1-document2"], d["bias2-document1"], d["bias2-document2"]]
        base_result, base_scores = get_retrieval_results(q, docs, mode="None")
        if mode == "decompose":
            results, scores = get_retrieval_results(q, docs, mode="decompose")
        elif mode == "rewrite":
            results, scores = get_retrieval_results(q, docs, mode="rewrite")
        elif mode == "both":
            results, scores = get_retrieval_results(q, docs, mode="both")
            
        base_embed = embed_documents(base_result)
        results_embed = embed_documents(results)
        # Calculate metrics
        overlap_scores.append(doc_overlap(base_result, results))
        sem_scores.append(sem_similarity(base_embed, results_embed))
        rep_variance_scores.append(representation_variance(results, group_set=group_set))

    # Save the results