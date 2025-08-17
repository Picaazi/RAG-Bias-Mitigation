# src/loaders/multi_dataset_loader.py
import os
import pandas as pd
from datasets import load_dataset
from src.db import get_connection

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def insert_docs(source, texts):
    clean_texts = [(source, t.strip()) for t in texts if isinstance(t, str) and t.strip()]
    if not clean_texts:
        return

    with get_connection() as conn:
        cur = conn.cursor()
        cur.executemany(
            "INSERT INTO documents (source, text) VALUES (?, ?)",
            clean_texts
        )
        conn.commit()

def load_gender_bias():
    return (
        pd.read_csv(os.path.join(BASE_DIR, "../../dataset/tasks/GenderBias-QA_train.csv")),
        pd.read_csv(os.path.join(BASE_DIR, "../../dataset/tasks/GenderBias-QA_test.csv"))
    )

def load_politics_bias():
    return (
        pd.read_csv(os.path.join(BASE_DIR, "../../dataset/tasks/PoliticBias-QA_train.csv")),
        pd.read_csv(os.path.join(BASE_DIR, "../../dataset/tasks/PoliticBias-QA_test.csv"))
    )

def load_bbq_datasets():
    bbq_dir = os.path.join(BASE_DIR, "../../BBQ/data")
    return (
        pd.read_json(os.path.join(bbq_dir, "Race_ethnicity.jsonl"), lines=True),
        pd.read_json(os.path.join(bbq_dir, "Religion.jsonl"), lines=True),
        pd.read_json(os.path.join(bbq_dir, "Gender_identity.jsonl"), lines=True),
        pd.read_json(os.path.join(bbq_dir, "Age.jsonl"), lines=True)
    )

def load_bibleqa():
    return pd.read_csv(os.path.join(BASE_DIR, "../../BibleQA/data/bible_qa/bible_qa_train.csv"))

def load_islamqa():
    dataset = load_dataset("minhalvp/islamqa", split="train")
    return dataset.to_pandas()

def load_all_docs_to_db():
    # GenderBias-QA
    gender_train, gender_test = load_gender_bias()
    insert_docs("GenderBias-QA", gender_train.get("context", []).dropna().tolist())
    insert_docs("GenderBias-QA", gender_test.get("context", []).dropna().tolist())

    # PoliticsBias-QA
    politics_train, politics_test = load_politics_bias()
    insert_docs("PoliticBias-QA", politics_train.get("context", []).dropna().tolist())
    insert_docs("PoliticBias-QA", politics_test.get("context", []).dropna().tolist())

    # BBQ datasets
    race_df, religion_df, genderid_df, age_df = load_bbq_datasets()
    insert_docs("BBQ-Race", race_df.get("context", []).dropna().tolist())
    insert_docs("BBQ-Religion", religion_df.get("context", []).dropna().tolist())
    insert_docs("BBQ-GenderIdentity", genderid_df.get("context", []).dropna().tolist())
    insert_docs("BBQ-Age", age_df.get("context", []).dropna().tolist())

    # BibleQA
    bible_df = load_bibleqa()
    insert_docs("BibleQA", bible_df.get("KJV_Verse", []).dropna().tolist())

    # IslamQA
    islam_df = load_islamqa()
    if "Full Answer" in islam_df.columns:
        insert_docs("IslamQA", islam_df["Full Answer"].dropna().tolist())
    else:
        print("IslamQA column 'Full Answer' not found. Skipping.")

    print("✅ All datasets loaded into SQLite!")
