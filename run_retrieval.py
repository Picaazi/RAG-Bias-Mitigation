# run_retrieval.py
import pandas as pd
from datasets import load_dataset
from src.embedders import Embedder
from src.retriever import Retriever

def main():
    method = "dense"  # or "bm25"
    model_name = "BAAI/bge-m3"
    cache_path = "embeddings.pkl"

    # Load dataset (IslamQA example)
    print("Loading dataset...")
    islamQA = load_dataset("minhalvp/islamqa", split="train").to_pandas()
    docs = islamQA["Full Answer"].dropna().tolist()

    print(f"Loaded {len(docs)} documents. Initializing embedder...")
    embedder = Embedder(model_name=model_name, cache_path=cache_path)

    # Load cached embeddings or create new ones
    embeddings = embedder.load_cache()
    if embeddings is None:
        print("No cache found. Encoding documents...")
        embeddings = embedder.encode(docs)
        embedder.cache_embeddings(embeddings)
    else:
        print("Loaded embeddings from cache.")

    retr = Retriever(docs, method=method, embedder=embedder, embeddings_cache=cache_path)

    print("\nReady for queries!")
    while True:
        q = input("Enter query (or 'exit'): ")
        if q.strip().lower() in ("exit", "quit"):
            break

        results = retr.retrieve(q, top_k=5)

        print("\nTop results:")
        for hit in results["results"]:
            print(f"{hit['rank']}. score={hit['score']:.4f} id={hit['doc_id']}")
            print(f"    {hit['doc'][:200]}...\n")

if __name__ == "__main__":
    main()
