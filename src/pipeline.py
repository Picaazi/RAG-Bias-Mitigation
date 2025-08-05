import data_reading
from decomposition import decompose_query

def run_pipeline():
    """
    Main pipeline function to run the entire process.
    """
    # Step 1: Load Data
    train, test = data_reading.read_gender_bias_data()
    print("Data loaded successfully.")
    print(f"Number of training samples: {len(train)}")
    print(f"Number of testing samples: {len(test)}")
    
    print(test.columns)

    for i in range(len(train)):
        d = train.iloc[i]
        q = d["question"]
        docs= [d["bias1-document1"], d["bias1-document2"], d["bias2-document1"], d["bias1-document2"]]
        
        sub_qs = decompose_query(d)

        ### classify which sub_qs are bias
        ### repharse bias sub_qs?
        ### retrieve document based on sub_qs
        ### Get final ans
        

if __name__ == "__main__":
    run_pipeline()