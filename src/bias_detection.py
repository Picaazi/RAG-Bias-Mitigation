from client import query_openai
def detect_bias(sub_questions):
                """Detect bias in a list of sub-questions using GPT-based classification."""
                bias_results = []
                
                for q in sub_questions:
                    bias_prompt = f"Analyze the following question for demographic bias (gender, race, age, etc.). Respond with 'BIASED' if it contains demographic bias, or 'NEUTRAL' if it doesn't: {q}"
                    bias_result = query_openai(bias_prompt)
                    
                    is_biased = bias_result and "BIASED" in bias_result.upper()
                    bias_results.append(is_biased)
                
                return bias_results