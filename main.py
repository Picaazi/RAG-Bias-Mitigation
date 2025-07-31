from src.decomposition import decompose_query
from src.client import query_openai
from src.rag_system import embed_documents
import argparse
import os


def detect_bias(sub_questions):
                """Detect bias in a list of sub-questions using GPT-based classification."""
                bias_results = []
                
                for q in sub_questions:
                    bias_prompt = f"Analyze the following question for demographic bias (gender, race, age, etc.). Respond with 'BIASED' if it contains demographic bias, or 'NEUTRAL' if it doesn't: {q}"
                    bias_result = query_openai(bias_prompt)
                    
                    is_biased = bias_result and "BIASED" in bias_result.upper()
                    bias_results.append(is_biased)
                
                return bias_results

def decompose():
    query = input("Enter your query: ")
    response = decompose_query(query)

    if response:
        # Detect and display bias classification
        bias_results = detect_bias(response)
        print("\nBias Classification:")
        for i, (q, is_biased) in enumerate(zip(response, bias_results), 1):
            if is_biased:
                print(f"    {i}. BIASED: {q}")
            else:
                print(f"    {i}. NEUTRAL: {q}")
                
def rag():
    """RAG functionality with document embedding and retrieval."""
    # Read documents (you may need to adjust the path/method based on your setup)
    documents = []
    try:
        # Example: reading from a file or directory
        # You'll need to implement document loading based on your requirements
        doc_path = input("Enter path to documents (or press Enter for default): ").strip()
        if doc_path:
            # Load documents from specified path
            with open(os.path.join(os.curdir, doc_path), 'r') as f:
                documents = [line.strip() for line in f.readlines() if line.strip()]
                
    except Exception as e:
        print(f"Error loading documents: {e}")
        return
    
    if not documents:
        print("No documents found.")
        return
    
    print(f"Loaded {len(documents)} documents")
    
    # Generate embeddings for the documents
    embeddings = embed_documents(documents)
    
    print(f"Generated embeddings for {len(embeddings)} documents")
    return embeddings

def idea1():
    query = input("Enter your query: ")
    response = decompose_query(query)
                
def main(mode):
    if mode == "decompose":
        decompose()
    elif mode == "rewrite":
        # Add rewrite functionality here
        pass
    elif mode == "rag":
        # Add RAG functionality here
        e = rag()
        for i in e:
            print(e[i])
        pass
    elif mode == 'idea1':
        idea1()
    else:
        print("Invalid mode. Use 'decompose', 'rewrite', or 'rag'.")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Bias Mitigation Tool")
    parser.add_argument("--mode", choices=["decompose", "rewrite", "rag"], required=True,
                       help="Mode of operation: decompose, rewrite, or rag")
    
    args = parser.parse_args()
    main(args.mode)



"""Rewrite/perturb query into neutral phrasing"""
def rewrite_query(query): 
    perturbed_queries = [ ]

    for q in query: 
        prompt = f"Rephrase the query to remove bias-inducing phrasing while maintaining the core contents"
        response = query_openai(prompt)
    if response: 
        perturbed_queries.append(response.strip())

    return perturbed_queries if perturbed_queries else None 
