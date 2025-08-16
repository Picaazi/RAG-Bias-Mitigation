import os
import re
from retrieval import retrieve_top_docs
from generation import generate_response

Bias_groups={
    "Race": ["Black", "White", "Asian", "Latino"],
    "Gender": ["Women", "Men", "Female", "Male","he","she"],
    "Politics": ["Conservative", "Democrat", "Republican","Left claim","Right "],
    "Religion": ["Christian", "Muslim", "Jewish", "Hindu"]
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
    top_docs = retrieve_top_docs(query)

    # Generating response
    print("\n--- LLM Response ---")
    response = generate_response(query, top_docs)
    print(response)

    # Extracting demographic mentions from response
    response_mentions = extract_demographic_mentions(response)

    # Extracting demographic mentions from retrieved documents
    docs_mentions = [extract_demographic_mentions(doc) for doc in top_docs]

    # Printing analysis
    print("\n--- Demographic Mentions in Response ---")
    print(response_mentions)

    print("\n--- Demographic Mentions in Retrieved Documents ---")
    for idx, mentions in enumerate(docs_mentions, 1):
        print(f"Document {idx}: {mentions}")
  
