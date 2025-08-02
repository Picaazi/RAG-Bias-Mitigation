from .client import query_openai

def decompose_query(query, num_subquestions=-1):
    """
    Decompose a complex query into smaller sub-questions.
    
    Args:
        query (str): The original query to decompose
        num_subquestions (int): Number of sub-questions to generate
    
    Returns:
        list: List of sub-questions
    """
    
    if num_subquestions == -1:
        prompt = f"""
        Decompose the following complex question into the minimum number of essential, non-redundant sub-questions that would help answer the original question comprehensively.
        Each sub-question should be distinct and not overlap with others.
        
        Original question: {query}

        Please provide each sub-question on a new line, numbered.
        """
    else:
        prompt = f"""
        Decompose the following complex question into {num_subquestions} smaller, more specific sub-questions that would help answer the original question comprehensively.
        Each sub-question should be distinct and not overlap with others.
        
        Original question: {query}
        
        Please provide exactly {num_subquestions} sub-questions, each on a new line, numbered 1-{num_subquestions}.
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
