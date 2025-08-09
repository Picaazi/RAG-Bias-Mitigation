import data_reading
from decomposition import decompose_query
from rewriting import rewrite_query
from bias_detection import detect_bias
import bm25s

def run_pipeline():
    """
    Main pipeline function to run the entire process.
    """
    # Step 1: Load Data
    train, test = data_reading.read_gender_bias_data()
    print("Data loaded successfully.")
    print(f"Number of training samples: {len(train)}")
    print(f"Number of testing samples: {len(test)}")
    

    for i in range(len(train)):
        d = train.iloc[i]
        q = d["question"]
        docs= [d["bias1-document1"], d["bias1-document2"], d["bias2-document1"], d["bias2-document2"]]
        
        sub_qs = decompose_query(d)
        sub_qs_biased = detect_bias(sub_qs)
        
        ### Only decomposition and only rewriting
        for j in range(len(sub_qs)):
            if sub_qs_biased[j]:
                print(f"Sub-question {j+1} is biased: {sub_qs[j]}")
                new_q = rewrite_query(sub_qs[j])
                print(f"Rephrased to neutral: {new_q}")
                sub_qs[j] = new_q[0]
                
            else:
                print(f"Sub-question {j+1} is neutral: {sub_qs[j]}")

        # Create the BM25 model and index the corpus
        retriever = bm25s.BM25(corpus=docs)
        retriever.index(bm25s.tokenize(docs))
            
        for j in range(len(sub_qs)):
            # Query the corpus and get top-k results
            results, scores = retriever.retrieve(bm25s.tokenize(sub_qs[j]), k=4) 

            # Let's see what we got!
            print(sub_qs[j])
            print(results)
            print(scores)

            if j >= 0:
                break
            
        ### Get final ans
        if i >= 0:
            break

if __name__ == "__main__":
    run_pipeline()