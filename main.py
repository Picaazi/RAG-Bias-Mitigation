from src.decomposition import decompose_query
from src.client import query_openai
from src.rag_system import embed_documents
import argparse
import os
import openai

openai.api_key = "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"

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

def generate_response(query, top_docs):
    context_text = "\n\n".join([f"{i+1}. {doc}" for i, doc in enumerate(top_docs)])
    prompt = f"""
You are a helpful assistant. Use the following context to answer the question accurately.

Question: {query}

Context:
{context_text}

Answer in detail:
"""
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a knowledgeable assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )
    return response.choices[0].message.content.strip()

