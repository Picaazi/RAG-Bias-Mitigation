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
    #!git clone https://github.com/danielkty/debiasing-rag
    genderbiasQA_train = pd.read_csv("debiasing-rag/dataset/tasks/GenderBias-QA_train.csv")
    genderbiasQA_test = pd.read_csv("debiasing-rag/dataset/tasks/GenderBias-QA_test.csv")
    politicbiasQA_train = pd.read_csv("debiasing-rag/dataset/tasks/PoliticBias-QA_train.csv")
    politicbiasQA_test = pd.read_csv("debiasing-rag/dataset/tasks/PoliticBias-QA_test.csv")

    #!git clone https://github.com/nyu-mll/BBQ
    race_BBQ=pd.read_json("BBQ/data/Race_ethnicity.jsonl", lines=True)
    religion_BBQ=pd.read_json("BBQ/data/Religion.jsonl", lines=True)
    genderidentity_BBQ=pd.read_json("BBQ/data/Gender_identity.jsonl", lines=True)
    age_BBQ=pd.read_json("BBQ/data/Age.jsonl", lines=True)

    #!git clone https://github.com/helen-jiahe-zhao/BibleQA
    bibleQA=pd.read_csv("BibleQA/data/bible_qa/bible_qa_train.csv", sep='\t')

    islamQA= load_dataset("minhalvp/islamqa", split="train").to_pandas()
    docs = islamQA["Full Answer"].dropna().tolist()+bibleQA["KJV_Verse"].dropna().tolist()+genderbiasQA_train["bias1-document1"].dropna().tolist()+genderbiasQA_train["bias1-document2"].dropna().tolist()+race_BBQ["context"].dropna().tolist()+religion_BBQ["context"].dropna().tolist()+\
#genderidentity_BBQ["context"].dropna().tolist()+\
#age_BBQ["context"].dropna().tolist()

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
