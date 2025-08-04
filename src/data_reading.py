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

if __name__ == "__main__":
    gender_train_df, gender_test_df = read_gender_bias_data()
    politics_train_df, politics_test_df = read_politics_bias_data()

    print(gender_train_df.columns)
    for i in range(5):
        print(gender_train_df.iloc[i])
