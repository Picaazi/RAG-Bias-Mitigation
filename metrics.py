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
    orig_embed: average embedding of top-k documents retrieved by original 
    new_embed: average embedding of top-k documents retrieved by reformed queries 
    """


