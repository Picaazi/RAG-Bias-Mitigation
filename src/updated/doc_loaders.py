import pandas as pd
import random
import sys
import os

# -----------------------------
# Setup project path
# -----------------------------
# Add the project root to Python path so we can import modules from other folders
SRC_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(SRC_ROOT)


# =============================
# Function: load_corpus_docs
# =============================
def load_corpus_docs(corpus_path):
    """
    Load corpus documents used as distractors.
    
    Parameters:
    - corpus_path (str): Path to CSV file containing corpus documents.
    
    Assumes the CSV has a column named 'text'.
    
    Returns:
    - List[str]: List of document texts (strings).
    """
    # Read CSV file into pandas DataFrame
    df = pd.read_csv(corpus_path)
    
    # Convert the 'text' column to a list of strings, drop empty values
    return [str(x) for x in df["text"].dropna().tolist()]


# =============================
# Function: build_genderbias_eval_index
# =============================
def build_genderbias_eval_index(
    csv_path,
    corpus_docs,
    n_distractors=500,
    seed=None
):
    """
    Build evaluation index for GenderBias-QA.

    Each question in the CSV is paired with:
    - base_docs: ground-truth relevant documents (bias1)
    - docs: retrieval pool = base_docs + randomly sampled distractors
    
    Parameters:
    - csv_path (str): Path to CSV containing test questions.
    - corpus_docs (List[str]): List of candidate distractor documents.
    - n_distractors (int): Number of distractors to sample per question.
    - seed (int, optional): Random seed for reproducibility.
    
    Returns:
    - dict: {qid: {"question": str, "docs": List[str], "base_docs": List[str]}}
    """
    # Read the CSV file with questions and base documents
    df = pd.read_csv(csv_path)
    
    eval_index = {}

    # Seed the random number generator for reproducibility
    if seed is not None:
        random.seed(seed)

    # Iterate over all questions in the CSV
    for i, row in df.iterrows():
        question = str(row["question"])

        # -----------------------------
        # Collect base docs (ground-truth)
        # -----------------------------
        base_docs = []
        for col in df.columns:
            if (col.startswith("bias1-document") or col.startswith("bias2-document")) and pd.notna(row[col]):
                base_docs.append(str(row[col]))


        # Skip questions with no base docs
        if not base_docs:
            continue

        # -----------------------------
        # Sample distractors
        # -----------------------------
        # Convert base docs to lowercase + strip whitespace for comparison
        base_set = set(d.lower().strip() for d in base_docs)
        
        # Keep only corpus docs that are NOT in the base set
        candidates = [
            d for d in corpus_docs
            if d.lower().strip() not in base_set
        ]

        # Skip if no candidates available
        if not candidates:
            continue

        # Randomly sample distractors (up to n_distractors)
        distractors = random.sample(
            candidates,
            min(n_distractors, len(candidates))
        )

        # -----------------------------
        # Add question to evaluation index
        # -----------------------------
        eval_index[i] = {
            "question": question,
            "docs": base_docs + distractors,   # retrieval corpus
            "base_docs": base_docs             # ground-truth relevant docs
        }

    return eval_index


# =============================
# Function: load_distractors
# =============================
def load_distractors(train_csv_path):
    """
    Extract all bias documents from GenderBias-QA_train.csv
    to be used as distractors.

    Parameters:
    - train_csv_path (str): Path to CSV containing training questions.
    
    Returns:
    - List[str]: List of unique bias documents from train set.
    """
    df = pd.read_csv(train_csv_path)

    docs = []
    for col in df.columns:
        # Collect all bias1 and bias2 documents
        if col.startswith("bias1-document") or col.startswith("bias2-document"):
            docs.extend(df[col].dropna().astype(str).tolist())

    # Remove duplicates
    docs = list(set(docs))

    return docs
