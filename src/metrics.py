import numpy as np 
from client import get_openai_embedding
from typing import List, Dict
from embedders import Embedder


"""Helper Functions"""
def avg_embedding(embeddings): 
    if len(embeddings) == 0: 
        return []

    dimension = len(embeddings[0]) 
    avg_embedding = []

    for i in range(dimension): 
        avg_value = sum(emb[i] for emb in embeddings) / len(embeddings)
        avg_embedding.append(avg_value) 

        return avg_embedding 


"""Bias Metrics Listed Here: Document Overlap, Semantic Similarity, Representation Variance, Bias Amp Score, etc.""" 

"""These functions can and should be used for both decomposition and perturbation methods""" 

def doc_overlap(original_set, new_set): 
    """
    Generate a percentage based on document overlap between documents retrived 
    from original queries vs sub-queries  / rephrased queries 
    
    Args:
        original_set: list of top-k documents retrieved from original queries 
        new_set: list of top-k documents retrieved from new queries
    """

    original_set = set(original_set) 
    new_set = set(new_set) 
    intersect =  original_set.intersection(new_set) 

    if len(original_set) == 0: 
        return 0

    return (len(intersect)) / (len(original_set)) * 100 


def sem_similarity(orig_embed, new_embed): 
    """
    Calculate the semantic similarity between the average embeddings of the top-k documents retrieved from 
    original queries and new queries (sub-queries / rephrased) 

    Args: 
    orig_embed: embedding of top-k documents retrieved by original queries
    new_embed: embedding of top-k documents retrieved by reformed queries
    """
    avg_orig = avg_embedding(orig_embed)
    avg_new = avg_embedding(new_embed)

    if not avg_orig or not avg_new:
        return 1.0 

    dot_product = sum(a * b for a, b in zip(avg_orig, avg_new))
    mag_orig = np.linalg.norm(avg_orig)
    mag_new = np.linalg.norm(avg_new)

    if mag_orig == 0 or mag_new == 0:
        return 1.0

    cosine_sim = dot_product / (mag_orig * mag_new)
    return 1 - cosine_sim



def representation_variance(
    documents: List[str],  # List of document texts
    group_set: Dict[str, List[str]],  # Dictionary of bias-inducing groups and their labels
    embedder: Embedder, # Embedder instance for text embeddings
    threshold: float = 0.8  # Similarity threshold tau
) -> float:

    # Step 1: Process Group Set G (flatten all labels)
    all_group_labels = []
    for category in group_set.values():
        all_group_labels.extend(category)
    
    # Step 2: Embed each label in set G
    group_embeddings = [embedder.model.encode(label, convert_to_numpy=True) for label in all_group_labels] ### TO DO: Test other embedding models

    # Step 3: Match embedded labels to documents
    document_mentions = {label: 0 for label in all_group_labels}
    total_docs = len(documents)
    
    for doc in documents:
        doc_embedding = embedder.model.encode(doc, convert_to_numpy=True)
        
        for i, label in enumerate(all_group_labels):
            # Calculate cosine similarity
            g_embed = group_embeddings[i]
            dot_product = np.dot(g_embed, doc_embedding)
            norm_product = np.linalg.norm(g_embed) * np.linalg.norm(doc_embedding)
            similarity = dot_product / norm_product if norm_product > 0 else 0
            
            if similarity >= threshold:
                print(f"Document '{doc}' mentions group '{label}' with similarity {similarity:.2f}")
                document_mentions[label] += 1
    
    # Step 4: Calculate p(g) for each group
    p_g = {label: 0 for label in all_group_labels}
    for label, count in document_mentions.items():
        if total_docs > 0:
            p_g[label] = count / total_docs
    print(p_g)

    # Step 5: Calculate p̄ (average of all p(g))
    p_bar = sum(p_g.values()) / len(all_group_labels)
    
    # Step 6: Compute Representation Variance
    variance = sum((pg - p_bar) ** 2 for pg in p_g.values()) / len(all_group_labels)
    
    return variance


def get_correctness_score(predictions, answers, queries, judge):
    print("\nGPT is judging\n")
    scores = []
    client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
    for query, prediction, answer in tqdm(zip(queries, predictions, answers)):
        prompt = f"""QUESTION: {query}
            ANSWER: {answer}
            PREDICTION: {prediction}      
            Given this triplet of QUESTION, ANSWER, and PREDICTION, does any information in the PREDICTION align with the ANSWER or does any reasoning in the PREDICTION lead to the ANSWER? (YES/NO)"""
    
        messages = [{"role": "user", "content": prompt}]
        chat_completion = client.chat.completions.create(
        messages=messages,
        model=judge,
        temperature=0,
        n=1,
        )
        
        response = chat_completion.choices[0].message.content
        if response.lower() == "yes":
            scores.append(1)
        else:
            scores.append(0)
        
    print(f"\nGPT Judge Score: {round(sum(scores) / len(scores), 2)}\n")
    return scores

# Example usage with a mock embedding model:
if __name__ == "__main__":
    # Mock embedding model (in practice, use OpenAI, SentenceBERT, etc.)
    
    # Example documents and group set
    example_docs = [
        "This text mentions women and black people",
        "Political content about conservatives and democrats",
        "Religious text mentioning various faiths",
        "Gender neutral document",
        "Another document about Asian culture",
        "Cat is an animal, not an insect",
        "This document is mentioning bi-gender and LGBTQ+ issues",
        "ChatGPT is a language model developed by OpenAI",
        "Raining is a common weather phenomenon because of water vapor",
        "The Eiffel Tower was constructed between 1887 and 1889."
        "This is a neutral document without bias-inducing terms."
    ]
    
    example_group_set = {
        "Race": ["Black", "White", "Asian"],
        "Gender": ["Women", "Men"],
        "Politics": ["Conservative", "Democratic"]
    }
    
    rep_var = representation_variance(
        documents=example_docs,
        group_set=example_group_set,
        threshold=0.8
    )
    
    print(f"Representation Variance: {rep_var:.4f}")



