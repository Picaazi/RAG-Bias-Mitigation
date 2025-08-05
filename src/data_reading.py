# git clone https://github.com/danielkty/debiasing-rag
import pandas as pd

def read_gender_bias_data():
    """
    Reads gender bias data from CSV files and returns the DataFrames.
    """
    gender_train_df = pd.read_csv("./debiasing-rag/dataset/tasks/GenderBias-QA_train.csv")
    gender_test_df = pd.read_csv("./debiasing-rag/dataset/tasks/GenderBias-QA_test.csv")
    return gender_train_df, gender_test_df

def read_politics_bias_data():
    """
    Reads politics bias data from CSV files and returns the DataFrames.
    """
    politics_train_df = pd.read_csv("./debiasing-rag/dataset/tasks/PoliticBias-QA_train.csv")
    politics_test_df = pd.read_csv("./debiasing-rag/dataset/tasks/PoliticBias-QA_test.csv")
    return politics_train_df, politics_test_df

def read_islamqa_data():
    import pandas as pd

    df = pd.read_parquet("hf://datasets/minhalvp/islamqa/data/train-00000-of-00001.parquet")
    return df

if __name__ == "__main__":
    gender_train_df, gender_test_df = read_gender_bias_data()
    politics_train_df, politics_test_df = read_politics_bias_data()
    islamqa_df = read_islamqa_data()

    print(gender_train_df.head())
