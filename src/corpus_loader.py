from datasets import load_dataset
import pandas as pd
import os
import json
import ir_datasets



# --- Output folder in repo ---
output_folder = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "corpus_data"
)
os.makedirs(output_folder, exist_ok=True)  

# --- Helper to flatten MS MARCO passages and remove duplicates ---
def flatten_and_deduplicate(dataset):
    rows = []
    for sample in dataset:
        query = sample['query']
        query_id = sample['query_id']
        for i, passage_text in enumerate(sample['passages']['passage_text']):
            is_selected = sample['passages']['is_selected'][i]
            url = sample['passages']['url'][i]
            rows.append({
                'query_id': query_id,
                'query': query,
                'passage_text': passage_text,
                'is_selected': is_selected,
                'url': url
            })
    df = pd.DataFrame(rows)
    # Normalize text for deduplication
    df['passage_text_norm'] = df['passage_text'].str.strip().str.lower()
    df = df.drop_duplicates(subset=['query_id', 'passage_text_norm'])
    df = df.drop(columns=['passage_text_norm'])
    return df

# # --- Load small subset for testing (adjust [:1%] for larger dataset) ---
# print("Downloading MS MARCO train split...")
# msmarco_train = load_dataset("ms_marco", "v2.1", split="train[:1%]")

# print("Downloading MS MARCO test split...")
# msmarco_test = load_dataset("ms_marco", "v2.1", split="test[:1%]")

# # --- Flatten and deduplicate ---
# print("Flattening and deduplicating train split...")
# train_df = flatten_and_deduplicate(msmarco_train)

# print("Flattening and deduplicating test split...")
# test_df = flatten_and_deduplicate(msmarco_test)

# # --- Save CSVs ---
# train_csv_path = os.path.join(output_folder, "msmarco_train.csv")
# test_csv_path = os.path.join(output_folder, "msmarco_test.csv")

# train_df.to_csv(train_csv_path, index=False)
# test_df.to_csv(test_csv_path, index=False)

# print(f"MS MARCO CSVs saved in {output_folder}!")

#################################################################################3

#Helper function to flatten FEVER dataset and remove duplicates 
def flatten_and_deduplicate_fever(dataset):
    rows = []
    for sample in dataset:
        query = sample['claim']            # FEVER claim
        query_id = sample['id']            # FEVER ID
        passages = sample.get('evidence', [])  # List of evidence
        for passage in passages:
            # Depending on your FEVER JSON, you may need to fetch text from each passage
            passage_text = " ".join([str(x) for x in passage])
            rows.append({
                'query_id': query_id,
                'query': query,
                'passage_text': passage_text
            })
    df = pd.DataFrame(rows)
    # Normalize text for deduplication
    df['passage_text_norm'] = df['passage_text'].str.strip().str.lower()
    df = df.drop_duplicates(subset=['query_id', 'passage_text_norm'])
    df = df.drop(columns=['passage_text_norm'])
    return df


##########################################################################################


# # Load FEVER corpus
# fever_json_path = os.path.join(output_folder, "fever_data.jsonl")
# dataset = []
# with open(fever_json_path, "r", encoding="utf-8") as f:
#     for line in f:
#         dataset.append(json.loads(line))

# train_df = flatten_and_deduplicate_fever(dataset)
# train_csv_path = os.path.join(output_folder, "fever_train.csv")
# train_df.to_csv(train_csv_path, index=False)

# print(f"FEVER CSV saved at {train_csv_path}")



############################################################### - edit 

##Natural Questions

# #subset 
# nq_train = load_dataset("natural_questions", "default", split="train[:1000]")
# df_train = pd.DataFrame(nq_train)
# output_folder = os.path.join(os.path.dirname(__file__), "../corpus_data")
# os.makedirs(output_folder, exist_ok=True)
# csv_path = os.path.join(output_folder, "nq_train.csv")
# df_train.to_csv(csv_path, index=False, encoding="utf-8")



#####################################################################################################################3



# # #Wikipedia - HuggingFace https://huggingface.co/datasets/wikimedia/wikipedia
# wiki_ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)

# # Collect 100 entries
# rows = []
# for i, article in enumerate(wiki_ds):
#     if i >= 100:  # limit to 100 entries
#         break
#     rows.append({
#         "id": article.get("id", ""),
#         "url": article.get("url", ""),
#         "title": article.get("title", ""),
#         "text": article.get("text", "")
#     })

# # Save to CSV
# df = pd.DataFrame(rows)
# csv_path = os.path.join(output_folder, "wiki_small.csv")
# df.to_csv(csv_path, index=False)

# print(f"Saved {len(rows)} wiki entries to {csv_path}")


###############################################################################################################



#PolNLI - https://huggingface.co/datasets/mlburnham/Pol_NLI/viewer/default/test?views%5B%5D=test
pol_nli_test = load_dataset("mlburnham/Pol_NLI", split="test")
df = pd.DataFrame(pol_nli_test)  
csv_path = os.path.join(output_folder, "pol_nli_test.csv")
df.to_csv(csv_path, index=False)  # no encoding needed
print(f"Saved {len(df)} entries to {csv_path}")
