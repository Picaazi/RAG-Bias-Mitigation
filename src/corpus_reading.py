import pandas as pd
import os 

CORPUS_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), "corpus_data")

def read_sbic_data():
    """
    Reads SBIC test data from CSV file and returns the DataFrame.
    """
    sbic_df = pd.read_csv(os.path.join(CORPUS_FOLDER, "SBIC.v2.tst.csv"))
    return sbic_df

def read_bbc_news_data():
    """
    Reads BBC News dataset from CSV file and returns the DataFrame.
    """
    bbc_df = pd.read_csv(os.path.join(CORPUS_FOLDER, "bbc_news.csv"))
    return bbc_df

def read_fever_train_data():
    """
    Reads FEVER training dataset from CSV file and returns the DataFrame.
    """
    fever_df = pd.read_csv(os.path.join(CORPUS_FOLDER, "fever_train.csv"))
    return fever_df

def read_msmarco_test_flat_data():
    """
    Reads MS MARCO flattened test dataset from CSV file and returns the DataFrame.
    """
    msmarco_df = pd.read_csv(os.path.join(CORPUS_FOLDER, "msmarco_test_flat.csv"))
    return msmarco_df

if __name__ == "__main__":
    sbic_df = read_sbic_data()
    bbc_df = read_bbc_news_data()
    fever_df = read_fever_train_data()
    msmarco_df = read_msmarco_test_flat_data()

    print(sbic_df.head())
    print(bbc_df.head())
    print(fever_df.head())

    print(msmarco_df.head())
