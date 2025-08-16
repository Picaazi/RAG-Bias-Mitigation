#pip install pyarrow
#git clone https://github.com/danielkty/debiasing-rag
import pandas as pd
from datasets import load_dataset

##Gender-biasQA##
gender_train_df = pd.read_csv("debiasing-rag/dataset/tasks/GenderBias-QA_train.csv")
gender_test_df = pd.read_csv("debiasing-rag/dataset/tasks/GenderBias-QA_test.csv")

##Politic-biasQA##
politics_train_df = pd.read_csv("debiasing-rag/dataset/tasks/PoliticBias-QA_train.csv")
politics_test_df = pd.read_csv("debiasing-rag/dataset/tasks/PoliticBias-QA_test.csv")

##BBQ dataset##
#git clone https://github.com/nyu-mll/BBQ
race_df=pd.read_json("BBQ/data/Race_ethnicity.jsonl", lines=True)
religion_df=pd.read_json("BBQ/data/Religion.jsonl", lines=True)
genderidentity_df=pd.read_json("BBQ/data/Gender_identity.jsonl", lines=True)
age_df=pd.read_json("BBQ/data/Age.jsonl", lines=True)

##BibleQA##
#git clone https://github.com/helen-jiahe-zhao/BibleQA
bibleQAtrain_df=pd.read_csv("BibleQA/data/bible_qa/bible_qa_train.csv")

##IslamQA##
dataset = load_dataset("minhalvp/islamqa", split="train")
df = dataset.to_pandas()
