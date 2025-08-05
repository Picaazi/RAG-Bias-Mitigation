from client import query_openai

"""Rewrite/perturb query into neutral phrasing"""
def rewrite_query(query): 
    perturbed_queries = []
    
    for q in query: 
        prompt = f"Rephrase the query to remove bias-inducing phrasing while maintaining the core contents and a neutral stance/viewpoint: {q}"
        response = query_openai(prompt)
        if response: 
            perturbed_queries.append(response.strip())
    
    return perturbed_queries