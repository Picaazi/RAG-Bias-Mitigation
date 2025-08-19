from src.client import query_openai
import openai

def generate_response(query, top_docs):
    context_text = "\n\n".join([f"{i+1}. {doc}" for i, doc in enumerate(top_docs)])
    prompt = f"""
You are a helpful assistant. Use the following context to answer the question accurately.

Question: {query}

Context:
{context_text}

Answer in detail:
"""
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a knowledgeable assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )
    return response.choices[0].message.content.strip()
