import data_reading
from decomposition import decompose_query, combine_queries
from rewriting import rewrite_query
from bias_detection import detect_bias
from metrics import doc_overlap, sem_similarity, representation_variance
import bm25s
import pandas as pd
import multi_dataset_loader as dataloader
import corpus_load_read
from retriever import Retriever
from embedders import Embedder
import time
import os 
from bias_grps import get_bias_grps
import openai

RESULTS_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
os.makedirs(RESULTS_FOLDER, exist_ok=True)

def extract_bias_groups(documents, save_csv=True, filename=None):
    bias_groups = get_bias_grps()
    results = []

    for i, doc in enumerate(documents):
        prompt = f"""
You are a bias group extractor.
The categories are: {list(bias_groups.keys())}
Each category has subgroups: {bias_groups}

Text: "{doc}"

Return a JSON object with categories as keys and detected subgroup mentions as lists.
Only include subgroups explicitly or implicitly present.
"""
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        try:
            extraction = response.choices[0].message["content"]
            extraction = pd.json_normalize([pd.json.loads(extraction)])
        except Exception:
            extraction = pd.DataFrame([{k: [] for k in bias_groups.keys()}])

        # Convert to long format for CSV
        for cat in bias_groups.keys():
            for subgroup in extraction.at[0, cat] if len(extraction) > 0 else []:
                results.append({
                    "doc_id": i,
                    "document": doc,
                    "category": cat,
                    "subgroup": subgroup
                })
        # If no subgroups detected, still include row
        if all(len(extraction.at[0, cat]) == 0 for cat in bias_groups.keys()):
            results.append({
                "doc_id": i,
                "document": doc,
                "category": "None",
                "subgroup": "None"
            })

    df = pd.DataFrame(results)

    if save_csv:
        timestamp = int(time.time())
        if filename is None:
            filename = f"llm_extraction_results_{timestamp}.csv"
        save_path = os.path.join(RESULTS_FOLDER, filename)
        df.to_csv(save_path, index=False)
        print(f"✅ LLM extraction results saved to {save_path}")

    return df
