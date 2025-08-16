# run_retrieval.py
#pip install pyarrow
#pip install openai
from datasets import load_dataset
import pandas as pd
import numpy as np
from rank_bm25 import BM25Okapi
from typing import List, Dict, Sequence
from FlagEmbedding import FlagModel
import os
import pickle
from sentence_transformers import SentenceTransformer
import openai
from collections import defaultdict
import re

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
    docs = islamQA["Full Answer"].dropna().tolist()+bibleQA["KJV_Verse"].dropna().tolist()+genderbiasQA_train["bias1-document1"].dropna().tolist()+genderbiasQA_train["bias1-document2"].dropna().tolist()+race_BBQ["context"].dropna().tolist()+religion_BBQ["context"].dropna().tolist()+genderidentity_BBQ["context"].dropna().tolist()+age_BBQ["context"].dropna().tolist()

   
    embedder = Embedder(model_name=model_name, use_flagmodel=True, cache_path=cache_path)
    retriever = Retriever(docs, method=method, embedder=embedder, embeddings_cache=cache_path)

    while True:
        q = input("Enter query (or 'exit'): ")
        if q.strip().lower() in ("exit", "quit"):
                break
        row = retriever.retrieve(q, top_k=5)
        row["meta"] = {"embedder_info": embedder.info() if embedder else None}

        #embed retrieved documents for analysis
        if embedder:
            retrieved_texts = [hit["doc"] for hit in row["results"]]
            retrieved_embeddings = embedder.encode_corpus(retrieved_texts)
            row["retrieved_embeddings"] = retrieved_embeddings.tolist()
        response=generate_response(q,retrieved_texts)
        bas=biasamplicationscore(retrieved_texts,response)
        print("\nTop results:")
        for hit in row["results"]:
          print(f"{hit['rank']}. score={hit['score']:.4f} id={hit['doc_id']}")
          print(f"    {hit['doc'][:200]}...\n")
        print("\nGenerated Response:")
        print(response)
        print("\nBias analysis:")
        for group, score in bas.items():
          print(f"{group}: {score:+.3f}"
                f" ( {'positive' if score>0 else 'negative' if score<0 else 'zero'})")
        print("\n"+ "="*80+"\n")



if __name__ == "__main__":
    main()
