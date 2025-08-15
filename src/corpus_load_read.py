
from datasets import load_dataset
import pandas as pd
import os
import json
import ir_datasets

#List of Corpuses 
#MS MARCO (DONE), WEBIS (stored FROM debiasing-rag REPO), FEVER (DONE), WIKIPEDIA (DONE, SBIC (DONE BUT CAN BE REFINED), BBC news (DONE),  NQ (done), 
#COMMON CRAWL / C4 (done) 

CORPUS_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), "corpus_data")
os.makedirs(CORPUS_FOLDER, exist_ok=True)

#Class for Wikipedia Corpus 
class Wikipedia: 
    #limit can be altered 
    def __init__(self, subset="20231101.en", limit=100):
        self.subset = subset; 
        self.limit = limit 
        self.file_path = os.path.join(CORPUS_FOLDER, "wiki_small.csv")

    #loads Wikipedia dataset with entries, saves as CSV 
    def process(self): 
        wiki_ds = load_dataset("wikimedia/wikipedia", self.subset, split="train", streaming=True)
        
        rows = []
        for i, article in enumerate(wiki_ds):
            if i >= self.limit:
                break
            rows.append({
                "id": article.get("id", ""),
                "url": article.get("url", ""),
                "title": article.get("title", ""),
                "text": article.get("text", "")
            })

        df = pd.DataFrame(rows)
        df.to_csv(self.file_path, index=False)
        print(f"Saved {len(rows)} Wikipedia entries to {self.file_path}")

    #returns csv as dataframe 
    def read(self): 
        """Read the saved Wikipedia CSV."""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"{self.file_path} not found. Make sure the CSV exists.")
        wiki_df = pd.read_csv(self.file_path)
        return wiki_df
    

#/////////////////////////////////////////////////////////////////////////////////////////////////////////////

#Class for PolNLI Corpus 
class PolNLI: 
    def __init__(self): 
        self.file_path = os.path.join(CORPUS_FOLDER, "pol_nli_test.csv")

    def process(self): 
        #downloads PolNli Test Split and saves as csv 
        pol_nli_test = load_dataset("mlburnham/Pol_NLI", split="test")
        df = pd.DataFrame(pol_nli_test)  
        df.to_csv(self.file_path, index=False) 
        print(f"Saved {len(df)} entries to {self.file_path}")

    def read(self): 
        #Read the saved csv after processing 
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"{self.file_path} not found. Make sure the CSV exists.")
        pol_nli_df = pd.read_csv(self.file_path)
        return pol_nli_df
    
#///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#Class for FEVER Corpus - FEVER JSON FILE HAS TO BE MANUALLY DOWNLOADED https://fever.ai/dataset/fever.html (training dataset)
class FEVER:
    def __init__(self, json_file="fever_data.jsonl"):
        self.json_file = os.path.join(CORPUS_FOLDER, json_file)
        self.file_path = os.path.join(CORPUS_FOLDER, "fever_train.csv")

    def flatten_and_deduplicate(self, dataset):
        """Flatten the FEVER dataset and remove duplicates."""
        rows = []
        for sample in dataset:
            query = sample['claim']
            query_id = sample['id']
            passages = sample.get('evidence', [])
            for passage in passages:
                passage_text = " ".join([str(x) for x in passage])
                rows.append({
                    'query_id': query_id,
                    'query': query,
                    'passage_text': passage_text
                })
        df = pd.DataFrame(rows)
        df['passage_text_norm'] = df['passage_text'].str.strip().str.lower()
        df = df.drop_duplicates(subset=['query_id', 'passage_text_norm'])
        df = df.drop(columns=['passage_text_norm'])
        return df

    def process(self):
        """Read FEVER JSONL, flatten, deduplicate, and save as CSV."""
        if not os.path.exists(self.json_file):
            raise FileNotFoundError(f"{self.json_file} not found. Place the FEVER JSONL in CORPUS_FOLDER.")
        
        dataset = []
        with open(self.json_file, "r", encoding="utf-8") as f:
            for line in f:
                dataset.append(json.loads(line))

        df = self.flatten_and_deduplicate(dataset)
        df.to_csv(self.file_path, index=False)
        print(f"FEVER CSV saved at {self.file_path}")

    def read(self):
        """Read the saved FEVER CSV."""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"{self.file_path} not found. Run `process()` first.")
        return pd.read_csv(self.file_path)

#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class MSMarcoDataset:
    def __init__(self, split="train", percent="1%"):
        """
        split: "train" or "test"
        percent: string slice for small subset, e.g., "1%" or "10%"
        """
        self.split = split
        self.percent = percent
        self.file_path = os.path.join(CORPUS_FOLDER, f"msmarco_{split}.csv")

    def flatten_and_deduplicate(self, dataset):
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
        # Normalize for deduplication
        df['passage_text_norm'] = df['passage_text'].str.strip().str.lower()
        df = df.drop_duplicates(subset=['query_id', 'passage_text_norm'])
        df = df.drop(columns=['passage_text_norm'])
        return df

    def process(self):
        """Download, flatten, deduplicate, and save MS MARCO CSV."""
        print(f"Downloading MS MARCO {self.split} split ({self.percent})...")
        msmarco_ds = load_dataset("ms_marco", "v2.1", split=f"{self.split}[:{self.percent}]")
        df = self.flatten_and_deduplicate(msmarco_ds)
        df.to_csv(self.file_path, index=False)
        print(f"Saved {len(df)} MS MARCO entries to {self.file_path}")

    def read(self):
        """Read the saved MS MARCO CSV."""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"{self.file_path} not found. Run `process()` first.")
        return pd.read_csv(self.file_path)

# /////////////////////////////////////////////////////////////////////////////////////////////////////////////

#Class for SBIC Corpus - FILES NEED TO BE DOWNLOADED MANUALLY https://maartensap.com/social-bias-frames/
#HuggingFace load_dataset seems to have errors 

class SBIC:
    def __init__(self):
        self.file_path = os.path.join(CORPUS_FOLDER, "SBIC.v2.tst.csv")
        self.display_columns = [
            "post", "offensiveYN", "intentYN", "sexYN", "sexReason",
            "targetCategory", "targetMinority", "targetStereotype", "whoTarget",
            "annotatorGender", "annotatorRace", "annotatorPolitics", "dataSource"
        ]

    def process(self):
        """Download the dataset and save as CSV."""
        dataset = load_dataset("allenai/social_bias_frames", split="train")
        df = pd.DataFrame(dataset)
        # Keep only the desired columns if they exist
        cols_to_use = [col for col in self.display_columns if col in df.columns]
        df[cols_to_use].to_csv(self.file_path, index=False)
        print(f"Saved {len(df)} entries to {self.file_path}")

    def read(self):
        """Read the SBIC CSV and return key columns with 'post' first."""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"{self.file_path} not found. Run `process()` first.")
        df = pd.read_csv(self.file_path)
        # Keep only columns that exist
        cols_to_use = [c for c in self.display_columns if c in df.columns]
        # Ensure 'post' is first
        if "post" in cols_to_use:
            cols_to_use.remove("post")
            cols_to_use = ["post"] + cols_to_use
        return df[cols_to_use]
    

#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#Class FOR BBC Corpus - CSV file can be downloaded from here: https://www.kaggle.com/datasets/gpreda/bbc-news

class BBC: 
    def __init__(self):
        self.file_path = os.path.join(CORPUS_FOLDER, "bbc_news.csv")

    def read(self):
        """Read the saved BBC CSV."""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"{self.file_path} not found.")
        return pd.read_csv(self.file_path)

# ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#Class for NQ Questions 

class NaturalQuestions:
    def __init__(self, split="train"):
        self.split = split
        self.file_path = os.path.join(CORPUS_FOLDER, f"nq_{split}.csv")

    def process(self, limit=None):
        """Download the dataset and save as CSV."""
        dataset = load_dataset("sentence-transformers/natural-questions", split=self.split)
        if limit:
            dataset = dataset.select(range(limit))
        df = pd.DataFrame(dataset)
        df.to_csv(self.file_path, index=False)
        print(f"Saved {len(df)} entries to {self.file_path}")

    def read(self):
        """Read the saved CSV from corpus folder."""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"{self.file_path} not found. Run `process()` first.")
        return pd.read_csv(self.file_path)
    

#Class for CommonCrawl / C4 (refined)


class C4Corpus:
    def __init__(self, limit=100):
         #limit (int): Number of entries to save for processing.
  
        self.limit = limit
        self.file_path = os.path.join(CORPUS_FOLDER, "c4_en_sample.csv")

    def process(self):
        """Download the English C4 dataset (streaming) and save a subset to CSV."""
        dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)
        rows = []
        for i, sample in enumerate(dataset):
            if i >= self.limit:
                break
            rows.append({"text": sample.get("text", ""), "timestamp": sample.get("timestamp", "")})
        
        df = pd.DataFrame(rows)
        df.to_csv(self.file_path, index=False)
        print(f"Saved {len(df)} C4 entries to {self.file_path}")

    def read(self):
        """Read the saved CSV from corpus folder."""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"{self.file_path} not found. Run `process()` first.")
        return pd.read_csv(self.file_path)

# Example 
if __name__ == "__main__":
    c4 = C4Corpus(limit=100)
    c4.process()
    df = c4.read()
    print(df.head())



