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

    intersect =  original_set.intersection(new_set) 
