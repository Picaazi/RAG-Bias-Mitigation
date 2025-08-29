# src/loaders/multi_dataset_loader.py
import pandas as pd
from datasets import load_dataset, load_from_disk, concatenate_datasets
from db import get_connection

def insert_docs(source, texts):
    with get_connection() as conn:
        cur = conn.cursor()
        cur.executemany(
            "INSERT INTO documents (source, text) VALUES (?, ?)",
            [(source, t) for t in texts if isinstance(t, str) and t.strip()]
        )
        conn.commit()

# New updates on genderbias and politicBias-QA dataset
class gender_bias():
    def __init__(self):
        self.file_path = "debiasing-rag/dataset/tasks/GenderBias-QA"
        self.dataset = load_from_disk(self.file_path)

    def query_and_ans(self):
        train = self.dataset["train"].to_pandas()
        test = self.dataset["test"].to_pandas()
        data = pd.concat([train, test])
        queries = data["queries"]
        answers = [data["bias1-document1"], data["bias1-document2"],data["bias2-document1"],data["bias2-document2"]]
        return queries
    
    def corpus(self):
        data = self.dataset["corpus"]
        corpus = data["text"]
        return corpus

class politics_bias():
    def __init__(self):
        self.file_path = "debiasing-rag/dataset/tasks/PoliticBias-QA"
        self.dataset = load_from_disk(self.file_path)

    def query_and_ans(self):
        train = self.dataset["train"]
        test = self.dataset["test"]
        val = self.dataset["val"]
        data = pd.concat([train, val, test])
        queries = data["queries"]
        answers = [data["left_claims"], data["right_claims"]]
        return queries, answers

    def corpus(self):
        data = self.dataset["corpus"]
        corpus = data["text"]
        return corpus

def load_gender_bias():
    return (
        pd.read_csv("debiasing-rag/dataset/tasks/GenderBias-QA_train.csv"),
        pd.read_csv("debiasing-rag/dataset/tasks/GenderBias-QA_test.csv")
    )

def load_politics_bias():
    return (
        pd.read_csv("debiasing-rag/dataset/tasks/PoliticBias-QA_train.csv"),
        pd.read_csv("debiasing-rag/dataset/tasks/PoliticBias-QA_test.csv")
    )

def load_bbq_datasets():
    return (
        pd.read_json("BBQ/data/Race_ethnicity.jsonl", lines=True),
        pd.read_json("BBQ/data/Religion.jsonl", lines=True),
        pd.read_json("BBQ/data/Gender_identity.jsonl", lines=True),
        pd.read_json("BBQ/data/Age.jsonl", lines=True)
    )

def load_bibleqa():
    return pd.read_csv("BibleQA/data/bible_qa/bible_qa_train.csv")

def load_islamqa():
    dataset = load_dataset("minhalvp/islamqa", split="train")
    return dataset.to_pandas()

def load_all_docs_to_db():
    # GenderBias-QA
    gender_train, gender_test = load_gender_bias()
    insert_docs("GenderBias-QA", gender_train["context"].dropna().tolist())
    insert_docs("GenderBias-QA", gender_test["context"].dropna().tolist())

    # PoliticsBias-QA
    politics_train, politics_test = load_politics_bias()
    insert_docs("PoliticBias-QA", politics_train["context"].dropna().tolist())
    insert_docs("PoliticBias-QA", politics_test["context"].dropna().tolist())

    # BBQ datasets
    race_df, religion_df, genderid_df, age_df = load_bbq_datasets()
    insert_docs("BBQ-Race", race_df["context"].dropna().tolist())
    insert_docs("BBQ-Religion", religion_df["context"].dropna().tolist())
    insert_docs("BBQ-GenderIdentity", genderid_df["context"].dropna().tolist())
    insert_docs("BBQ-Age", age_df["context"].dropna().tolist())

    # BibleQA
    bible_df = load_bibleqa()
    insert_docs("BibleQA", bible_df["KJV_Verse"].dropna().tolist())

    # IslamQA
    islam_df = load_islamqa()
    insert_docs("IslamQA", islam_df["Full Answer"].dropna().tolist())

    print("All datasets loaded into SQLite!")

if __name__ == "__main__":
    print(load_islamqa())
