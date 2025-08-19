import openai
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def query_openai(query, model="gpt-3.5-turbo"):
    """
    Call the OpenAI API with the given query and return the response.
    Requires the OPENAI_KEY environment variable to be set.
    """
    api_key = os.environ.get("OPENAI_KEY") #Use the command "export OPENAI_KEY='ur-api-key'" in terminal
    if not api_key:
        print("OPENAI_KEY environment variable not set.")
        return None
        
    # Initialize OpenAI client with the new v1.0+ interface
    client = openai.OpenAI(api_key=api_key)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": query}],
            temperature=0.7,
            max_tokens=512,
        )
        
        answer = response.choices[0].message.content.strip()
        return answer
    except Exception as e:
        print(f"Error querying OpenAI: {e}")
        return None

def get_openai_embedding(text, model="text-embedding-ada-002"):
    """
    Get OpenAI embedding for the given text.
    Requires the OPENAI_KEY environment variable to be set.
    """
    api_key = os.environ.get("OPENAI_KEY")
    if not api_key:
        print("OPENAI_KEY environment variable not set.")
        return None
        
    # Initialize OpenAI client
    client = openai.OpenAI(api_key=api_key)
    
    try:
        response = client.embeddings.create(
            model=model,
            input=text
        )
        
        embedding = response.data[0].embedding
        return embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return None