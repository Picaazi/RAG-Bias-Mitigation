import os
import re
from src.retrieval.retriever import Retriever
from src.embedders.embedders import Embedder
from src.generate_response import generate_response 

Bias_groups = {
    "Race": ["Black", "White","Mixed","Asian", "Latino","Middle-Eastern","Hispanic","Indigneous"],
    "Gender": ["Women", "Men", "Female", "Male","He","She","They","Transgender","Queer","Non-binary"],
    "Politics": ["Conservative", "Democrat", "Republican","Labour","Left claim","Right claim","Communist","Socialist"],
    "Religion": ["Christian", "Muslim", "Jewish", "Hindu","Buddhism","Jainism"]
    "Age":["Young","Old","Teenager","Young adult","Middle-aged","Elderly","Seniors","Juniors","Infant"]
}

def extract_demographic_mentions(text):
    mentions = {category: [] for category in Bias_groups}
    for category, keywords in Bias_groups.items():
        for word in keywords:
            if re.search(rf"\b{word}\b", text, re.IGNORECASE):
                mentions[category].append(word)
    return mentions

# Example usage
response = "The elderly man and a young Muslim woman were at the event."
print(extract_demographic_mentions(response))

if __name__ == "__main__":
    query = input("Enter your question: ")

    # Retrieve top documents
    top_docs = retrieve(query)

    # Generating response
    print("\n--- LLM Response ---")
    response = generate_response(query, top_docs)
    print(response)

    # Extracting demographic mentions from response
    response_mentions = extract_demographic_mentions(response)

    # Extracting demographic mentions from retrieved documents
    docs_mentions = [extract_demographic_mentions(docs) for doc in top_docs]

    # Printing analysis
    print("\n--- Demographic Mentions in Response ---")
    print(response_mentions)

    print("\n--- Demographic Mentions in Retrieved Documents ---")
    for idx, mentions in enumerate(docs_mentions, 1):
        print(f"Document {idx}: {mentions}")
  
