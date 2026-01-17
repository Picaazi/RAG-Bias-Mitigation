import os
import time
import pandas as pd
from dotenv import load_dotenv
import openai

env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "api.env"))
load_dotenv(env_path)
openai.api_key = os.environ.get("OPENAI_KEY")

if openai.api_key is None:
    raise ValueError("OPENAI_KEY not found in api.env")
print("OPENAI_KEY loaded successfully")


from doc_loaders import build_genderbias_eval_index, load_distractors
from new_pipeline import llm_pipeline_eval

# ===========================
# Configuration
# ===========================

# Paths to test set CSVs
GENDERBIAS_CSV = "C:\\Users\\Dylan Kim\\Desktop\\clone\\RAG-Bias-Mitigation\\debiasing-rag\\dataset\\tasks\\GenderBias-QA_test.csv"
POLITICALBIAS_CSV = "C:\\Users\\Dylan Kim\\Desktop\\clone\\RAG-Bias-Mitigation\\debiasing-rag\\dataset\\tasks\\PoliticBias-QA_test.csv"


# Retriever / pipeline settings
TOP_K = 5          # k for Recall@k
DISTRACTORS = 150  # number of distractors per query
SEED = 42

MODES = ["rewrite", "decompose", "both"]  # experiment modes

# Number of questions to use (set during evaluation)
NUM_QUESTIONS = 5 # None = use all, or an int for limited questions

# ===========================
# Helper function
# ===========================

def run_eval(csv_path, dataset_name, mode, max_questions=None):
    print(f"\n=== Running {dataset_name} | mode={mode} ===")

    # Load corpus docs as distractors
    corpus_docs = load_distractors(csv_path)

    # Build evaluation index (base + distractors)
    eval_index = build_genderbias_eval_index(
        csv_path,
        corpus_docs,
        n_distractors=DISTRACTORS,
        seed=SEED
    )

    # Limit number of questions if requested
    if max_questions is not None:
        eval_index = dict(list(eval_index.items())[:max_questions])

    print(f"Loaded {len(eval_index)} questions for evaluation")

    # Run the LLM evaluation pipeline
    start_time = time.time()
    df_metrics = llm_pipeline_eval(
        eval_index=eval_index,
        k=TOP_K,
        mode=mode
    )
    end_time = time.time()

    elapsed = end_time - start_time
    print(f"✅ Completed {dataset_name} | mode={mode} in {elapsed:.1f}s")

    # Save summary
    summary = {
    "dataset": dataset_name,
    "mode": mode,
    "top_k": TOP_K,
    "# questions": len(eval_index),
    "recall@k_avg": df_metrics["recall@k_sem"].mean(),
    "mrr_avg": df_metrics["mrr_sem"].mean(),
    "soft_overlap_avg": df_metrics["soft_overlap"].mean(),  # this one seems unchanged
    "time_sec": elapsed
}

    return summary, df_metrics

# ===========================
# Run experiments
# ===========================

all_summaries = []

for mode in MODES:
    # GenderBias-QA
    s, _ = run_eval(GENDERBIAS_CSV, "GenderBias-QA", mode, max_questions=NUM_QUESTIONS)
    all_summaries.append(s)

    # PoliticalBias-QA
    #, _ = run_eval(POLITICALBIAS_CSV, "PoliticalBias-QA", mode, max_questions=NUM_QUESTIONS)
   #all_summaries.append(s)

# ===========================
# Save summary table
# ===========================

summary_df = pd.DataFrame(all_summaries)
summary_path = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "results",
    f"eval_summary_{int(time.time())}.csv"
)
summary_df.to_csv(summary_path, index=False)
print(f"\n✅ All summary results saved to: {summary_path}")
