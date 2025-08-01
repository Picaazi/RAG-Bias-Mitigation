import numpy as np 

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

    if len(original_set) = 0: 
        return 0

return (len(intersect)) / (len(original_set)) * 100 


def sem_similarity(orig_embed, new_embed): 
    """
    Calculate the semantic similarity between the avereage embeddings of the top-k documents retrived from 
    original queries and new queries (sub-queries / rephrased) 

    Args: 
    orig_embed: embedding of top-k documents retrieved by original 
    new_embed: embedding of top-k documents retrieved by reformed queries 
    """
    avg_orig = avg_embedding(orig_embed)
    avg_new = avg_embedding(new_embed)

    if not avg_orig or not avg_new:
        return 1.0 

    avg_product = sum(a * b for a, b in zip(avg_orig, avg_new))
    mag_orig = magnitude(avg_orig)
    mag_new = magnitude(avg_new)

    if mag_orig == 0 or mag_new == 0:
        return 1.0

    cosine_sim = dot_product / (mag_orig * mag_new)
    return 1 - cosine_sim

def rep_var #Edits coming soon 

"""Extra Helper Functions"""
def avg_embedding(embeddings): 
    if not embeddings: 
        return []

dimension = len(embeddings[0]) 
avg_embedding = []

for i in range(dimension): 
    avg_value = sum(emb[i] for emb in embeddings) / len(embeddings)
    avg_embedding.append(avg_value) 

return avg_embedding 

def magnitude(vec): 
    return sum(x * x for x in vector) ** 0.5 


    


