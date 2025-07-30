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
    api_key = os.getenv("OPENAI_KEY")
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
