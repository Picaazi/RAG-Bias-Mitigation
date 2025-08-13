# run_retrieval.py
import pandas as pd
from datasets import load_dataset
from src.embedders import Embedder
from src.retriever import Retriever
from src.bias_scorer import BiasScorer

def main():
    method = "dense"
    model_name = "BAAI/bge-m3"
    cache_path = "embeddings.pkl"

    # Load example dataset
    print("Loading dataset...")
    islamQA = load_dataset("minhalvp/islamqa", split="train").to_pandas()
    docs = islamQA["Full Answer"].dropna().tolist()

    print(f"Loaded {len(docs)} documents. Initializing embedder...")
    embedder = Embedder(model_name=model_name, use_flagmodel=True, cache_path=cache_path)
    retr = Retriever(docs, method=method, embedder=embedder, embeddings_cache=cache_path)

    scorer = BiasScorer()

    print("Ready for queries!")
    while True:
        q = input("Enter query (or 'exit'): ")
        if q.strip().lower() in ("exit", "quit"):
            break

        # Retrieve docs
        result = retr.retrieve(q, top_k=5)
        top_docs = [hit["doc"] for hit in result["results"]]

        # Simulate response (you can later replace with actual model output)
        response = f"This is a placeholder response for query: {q}"

        # Calculate bias score
        bias_scores = scorer.score(response, top_docs)

        print("\nQuery:", q)
        print("\nResponse:", response)
        print("\nTop results:")
        for hit in result["results"]:
            print(f"{hit['rank']}. score={hit['score']:.4f} id={hit['doc_id']}")
            print(f"    {hit['doc'][:200]}...\n")

        print("\nBias Scores:", bias_scores)
        print("-" * 60)

if __name__ == "__main__":
    main()
