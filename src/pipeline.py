import data_reading
from decomposition import decompose_query
from rewriting import rewrite_query
from bias_detection import detect_bias
from metrics import doc_overlap, sem_similarity, representation_variance
import bm25s
import pandas as pd
from client import get_openai_embedding
from retriever import Retriever
from embedders import Embedder

def get_retrieval_results(question, docs, k=4, mode="None"):
    # Use BM25 as an example for retrieval
    retriever = bm25s.BM25(corpus=docs)
    retriever.index(bm25s.tokenize(docs))
    questions = []
    if mode == "decompose":
        print("Decomposed sub-queries:")
        sub_qs = decompose_query(question)
        for j, sub_q in enumerate(sub_qs):
            print(f"Sub-query {j+1}: {sub_q}")
        results, scores = retriever.retrieve(bm25s.tokenize(sub_qs), k=k)
        questions.extend(sub_qs)
    elif mode == "rewrite":
        print("Rewriting sub-queries:")
        new_q = rewrite_query(question)
        print(f"Original query: {question}")
        print(f"Rephrased query: {new_q[0]}")
        results, scores = retriever.retrieve(bm25s.tokenize(new_q), k=k)
        questions.extend(new_q)
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
        questions.extend(sub_qs)
    elif mode == "None":
        print("No modification will be made.")
        results, scores = retriever.retrieve(bm25s.tokenize(q), k=k)
        questions.append(q)
    else:
        raise ValueError("Invalid mode. Choose from 'decompose', 'rewrite', 'both', or 'None'.")

    for result, score in zip(results, scores):
            print(f"    {result} (Score: {score})")           
    return questions, results, scores

def pipeline(questions, docs, k=5, mode="None"):
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
        
        if mode == "decompose":
            print("Decomposed sub-queries:")
            sub_qs = decompose_query(q)
            for j, sub_q in enumerate(sub_qs):
                print(f"Sub-query {j+1}: {sub_q}")
            result = retriever.retrieve(query=sub_qs, top_k=k)
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
            result = retriever.retrieve(query=sub_qs, top_k=k)
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
    pipeline(train_questions, train_docs, mode="None")