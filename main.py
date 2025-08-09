from src.decomposition import decompose_query
from src.client import query_openai
from src.rag_system import embed_documents
import argparse
import os



def decompose():
    query = input("Enter your query: ")
        
    response = decompose_query(query, num_subqs=-1)

    if response:
        # Detect and display bias classification
        bias_results = detect_bias(response)
        print("\nBias Classification:")
        for i, (q, is_biased) in enumerate(zip(response, bias_results), 1):
            if is_biased:
                print(f"    {i}. BIASED: {q}")
            else:
                print(f"    {i}. NEUTRAL: {q}")

