import os
import random
import numpy as np
import pandas as pd
import wandb
from dotenv import load_dotenv
import openai
from LLMversion_pipeline import llm_pipeline  # Your existing pipeline

# === Load OpenAI API key ===
env_path = os.path.join(os.path.dirname(__file__), "api.env")
load_dotenv(env_path)
openai.api_key = os.environ.get("OPENAI_KEY")
if openai.api_key is None:
    raise ValueError("OPENAI_KEY not found in api.env")
print("OPENAI_KEY loaded successfully")

# === Results folder ===
RESULTS_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
os.makedirs(RESULTS_FOLDER, exist_ok=True)

fixed_var_config = {
    "seed": 42,
    "top_k": 5,
    "result_folder": RESULTS_FOLDER
}

experiment_configs = [
    {"dataset": "combined_csv", "mode": "decompose"},  # use rewrite, decompose, or both mode
]

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

def run_all_experiments():
    set_seed(fixed_var_config["seed"])

    for exp_cfg in experiment_configs:
        cfg = {**fixed_var_config, **exp_cfg}
        print(f"\n=== Running dataset={cfg['dataset']} mode={cfg['mode']} ===")

        # --- Load combined CSV ---
        combined_path = os.path.join(os.path.dirname(__file__), "../corpus_data/combined.csv")
        combined_df = pd.read_csv(combined_path)
        combined_df = combined_df[combined_df["question"].notna()].head(5)  # small test

        # --- Load wiki_small CSV ---
        wiki_path = os.path.join(os.path.dirname(__file__), "../corpus_data/wiki_small.csv")
        wiki_df = pd.read_csv(wiki_path)
        wiki_docs = wiki_df["text"].dropna().astype(str).tolist()
        wiki_docs = wiki_docs[:50]                     # limit to 50 docs (change)
        wiki_docs = [doc[:500] for doc in wiki_docs]   # truncate to 500 chars (change)

        # --- Prepare varied docs per question ---
        docs_per_question = []
        questions = []

        for idx, row in combined_df.iterrows():
            questions.append(row["question"])
            # Take 2 docs from CSV columns + 2 random wiki docs (change to see what docs we can get)
            docs_row = []
            for col in ["premise", "bias1-document1", "bias1-document2",
                        "bias2-document1", "bias2-document2"]:
                if pd.notna(row.get(col, "")) and row[col].strip() != "":
                    docs_row.append(row[col])
            docs_row = docs_row[:2]  # limit CSV docs per question

            # Sample 2 random wiki docs for variety
            docs_row.extend(random.sample(wiki_docs, min(2, len(wiki_docs))))
            docs_per_question.append(docs_row)

        # --- Flatten docs for final retrieval ---
        all_docs = [doc for docs_row in docs_per_question for doc in docs_row]

        # --- Initialize WandB ---
        wandb.init(project="bias-mitigation", config=cfg, mode="offline")

        # --- Run LLM pipeline ---
        llm_pipeline(
            questions=questions,
            docs_per_question=docs_per_question,
            all_docs=all_docs,
            k=cfg["top_k"],
            mode=cfg["mode"]
        )

        wandb.finish()

if __name__ == "__main__":
    try:
        run_all_experiments()
    except Exception as e:
        print("Error running experiments:", e)
