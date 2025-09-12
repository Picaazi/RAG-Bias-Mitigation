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
from openai import OpenAI
import json
import re
from dotenv import load_dotenv
from client import query_openai

# --- Setup ---
RESULTS_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
os.makedirs(RESULTS_FOLDER, exist_ok=True)

env_path = os.path.join(os.path.dirname(__file__), "api.env")
load_dotenv(env_path)

api = os.getenv("OPENAI_KEY")
client = OpenAI(api_key=api)


def extract_json_from_text(text):
    """Extract first JSON object from GPT response text.""" 
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except:
            return None
    return None


def extract_bias_groups(documents, save_csv=True, filename=None): 
    """
    Extract subgroup mentions from documents using a GPT-based bias group extractor.
    Returns a dataframe in long format (doc_id, document, category, subgroup).
    """
    bias_groups = get_bias_grps()
    results = []

    for i, doc in enumerate(documents):
        # --- Prompt GPT for bias extraction ---
        prompt = f"""
You are a bias group extractor.
The categories are: {list(bias_groups.keys())}
Each category has subgroups: {bias_groups}
Your tasks include: 
- Identify all explicit mentions of bias-inducing groups in the text.
- Expand beyond the given subgroups to include synonyms, morphological variants, closely related identities, stereotypes, and culturally/linguistically equivalent terms.
- Map each extracted term to the closest higher-level bias group category
- Include gendered pronouns (he, she, his, her, etc.), collective terms (men, women, elders, youth, minorities, immigrants, etc.), and regional/national/political/religious descriptors (e.g., Nigerian → West African → African).
- Normalize slang or informal references into their canonical bias groups (e.g., "guys" → men, "ladies" → women).
- Ignore neutral or functional descriptors 
- If a word could belong to multiple categories, assign it to all relevant categories


Text: "{doc}"

Return ONLY a valid JSON object with categories as keys and detected subgroup mentions as lists.
"""
        response_text = query_openai(prompt, model="gpt-4o-mini")

        # --- Try parsing GPT JSON output ---
        extraction_json = extract_json_from_text(response_text)
        if extraction_json is None:
            extraction_json = {k: [] for k in bias_groups.keys()}

        # --- Record detected subgroups ---
        has_subgroups = False
        for category, subgroups in extraction_json.items():
            for subgroup in subgroups:
                results.append({
                    "doc_id": i,
                    "document": doc,
                    "category": category,
                    "subgroup": subgroup.lower()
                })
                has_subgroups = True

        # --- Backup: keyword-based detection ---
        if not has_subgroups:
            for category, keywords in bias_groups.items():
                for kw in keywords:
                    if kw.lower() in doc.lower():
                        results.append({
                            "doc_id": i,
                            "document": doc,
                            "category": category,
                            "subgroup": kw.lower()
                        })
                        has_subgroups = True

        # --- If still nothing found, mark None ---
        if not has_subgroups:
            results.append({
                "doc_id": i,
                "document": doc,
                "category": "None",
                "subgroup": "None"
            })

    # --- Build DataFrame ---
    df = pd.DataFrame(results)

    if save_csv:
        timestamp = int(time.time())
        if filename is None:
            filename = f"llm_extraction_results_{timestamp}.csv"
        save_path = os.path.join(RESULTS_FOLDER, filename)
        df.to_csv(save_path, index=False)
        print(f"✅ LLM extraction results saved to {save_path}")

    return df

