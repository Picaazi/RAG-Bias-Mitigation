import pandas as pd
from datasets import load_dataset
from src.embedders import Embedder
from src.retriever import Retriever

def main():
    method = "dense"
    model_name = "BAAI/bge-m3"
    cache_path = "embeddings.pkl"

    # Load example dataset (IslamQA)
    print("Loading dataset...")
    islamQA = load_dataset("minhalvp/islamqa", split="train").to_pandas()
    docs = islamQA["Full Answer"].dropna().tolist()

    print(f"Loaded {len(docs)} documents. Initializing embedder...")
    embedder = Embedder(model_name=model_name, use_flagmodel=True, cache_path=cache_path)
    retr = Retriever(docs, method=method, embedder=embedder, embeddings_cache=cache_path)

    print("Ready for queries!")
    while True:
        q = input("Enter query (or 'exit'): ")
        if q.strip().lower() in ("exit", "quit"):
            break
        row = retr.retrieve(q, top_k=5)
        row["meta"] = {"embedder_info": embedder.info()}

        print("\nTop results:")
        for hit in row["results"]:
            print(f"{hit['rank']}. score={hit['score']:.4f} id={hit['doc_id']}")
            print(f"    {hit['doc'][:200]}...\n")

if __name__ == "__main__":
    main()
