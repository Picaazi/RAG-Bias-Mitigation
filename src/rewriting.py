from client import query_openai

"""Rewrite/perturb query into neutral phrasing"""
def rewrite_query(query): 
    perturbed_queries = []
    
    for q in query: 
        prompt = f"Rephrase the query to remove bias-inducing or assumption-laden language/phrasing while maintaining the original intents, core contents, and neutral stance/viewpoint: {q}"
        response = query_openai(prompt)
        if response: 
            perturbed_queries.append(response.strip())
    
    return perturbed_queries
