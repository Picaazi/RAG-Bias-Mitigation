#pip install pyarrow
#git clone https://github.com/danielkty/debiasing-rag
import pandas as pd
from datasets import load_dataset

print("Listing contents of GenderBias-QA directory:")
!ls debiasing-rag/dataset/tasks/GenderBias-QA
gender_train_df = pd.read_csv("debiasing-rag/dataset/tasks/GenderBias-QA_train.csv")
gender_test_df = pd.read_csv("debiasing-rag/dataset/tasks/GenderBias-QA_test.csv")

print("Listing contents of PoliticBias-QA directory:")
!ls debiasing-rag/dataset/tasks/PoliticBias-QA
politics_train_df = pd.read_csv("debiasing-rag/dataset/tasks/PoliticBias-QA_train.csv")
politics_test_df = pd.read_csv("debiasing-rag/dataset/tasks/PoliticBias-QA_test.csv")

#BBQ dataset
race_df=pd.read_json("BBQ/data/Race_ethnicity.jsonl", lines=True)
religion_df=pd.read_json("BBQ/data/Religion.jsonl", lines=True)
genderidentity_df=pd.read_json("BBQ/data/Gender_identity.jsonl", lines=True)
age_df=pd.read_json("BBQ/data/Age.jsonl", lines=True)

#git clone https://github.com/helen-jiahe-zhao/BibleQA
bibleQAtrain_df=pd.read_csv("BibleQA/data/bible_qa/bible_qa_train.csv")
dataset = load_dataset("minhalvp/islamqa", split="train")
df = dataset.to_pandas()



