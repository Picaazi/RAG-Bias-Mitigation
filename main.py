from src.decomposition import decompose_query
from src.client import query_openai


def detect_bias(sub_questions):
                """Detect bias in a list of sub-questions using GPT-based classification."""
                bias_results = []
                
                for q in sub_questions:
                    bias_prompt = f"Analyze the following question for demographic bias (gender, race, age, etc.). Respond with 'BIASED' if it contains demographic bias, or 'NEUTRAL' if it doesn't: {q}"
                    bias_result = query_openai(bias_prompt)
                    
                    is_biased = bias_result and "BIASED" in bias_result.upper()
                    bias_results.append(is_biased)
                
                return bias_results

if __name__ == "__main__":
    query = input("Enter your query: ")
    response = decompose_query(query)

    if response:
        # Detect and display bias classification
        bias_results = detect_bias(response)
        print("\nBias Classification:")
        for i, (q, is_biased) in enumerate(zip(response, bias_results), 1):
            if is_biased:
                print(f"    {i}. ⚠️  BIASED: {q}")
            else:
                print(f"    {i}. ✅ NEUTRAL: {q}")