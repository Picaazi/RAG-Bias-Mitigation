from client import query_openai

def decompose_query(query: str, num_subqs=-1):
    """
    Decompose a complex query into smaller sub-questions.
    
    Args:
        query (str): The original query to decompose
        num_subquestions (int): Number of sub-questions to generate
    
    Returns:
        list: List of sub-questions
    """
    
    if num_subqs == -1:
        prompt = f"""
        Decompose the following complex question into the minimum number of essential, non-redundant sub-questions that would help answer the original question comprehensively.
        Each sub-question should be distinct and not overlap with others.
        
        Original question: {query}

        Please provide each sub-question on a new line, numbered.
        """
    else:
        prompt = f"""
        Decompose the following complex question into {num_subqs} smaller, more specific sub-questions that would help answer the original question comprehensively.
        Each sub-question should be distinct and not overlap with others.
        
        Original question: {query}
        
        Please provide exactly {num_subqs} sub-questions, each on a new line, numbered 1-{num_subqs}.
        """

    response = query_openai(prompt)
    
    if not response:
        print("Error: No response received")
        return []
    
    # Parse the response to extract sub-questions
    lines = response.strip().split('\n')
    sub_questions = []
    
    for line in lines:
        line = line.strip()
        if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
            # Remove numbering and clean up
            question = line.split('.', 1)[-1].strip()
            if question:
                sub_questions.append(question)

    return sub_questions[:len(sub_questions)]

def combine_queries(sub_qs: List[str]):
    """
    Combine a list of sub-questions into a single query.
    
    Args:
        sub_qs (List[str]): The list of sub-questions to combine
    
    Returns:
        str: The combined query
    """
    
    combine_queries = f"""
    Answer the following sub-questions:

    {chr(10).join(f"{i+1}. {query}" for i, query in enumerate(sub_qs))}
    """
    return combine_queries
