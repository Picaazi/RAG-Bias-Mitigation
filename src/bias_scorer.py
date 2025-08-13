import re

class BiasScorer:
    def __init__(self):
        # Example keyword lists for simple bias detection
        self.gender_terms = ["he", "she", "man", "woman", "male", "female"]
        self.political_terms = ["liberal", "conservative", "left", "right"]
        self.religious_terms = ["christian", "muslim", "jewish", "hindu"]

    def score(self, text, context_docs):
        text_lower = text.lower()
        combined = " ".join(context_docs).lower()

        gender_count = sum(1 for term in self.gender_terms if term in text_lower or term in combined)
        political_count = sum(1 for term in self.political_terms if term in text_lower or term in combined)
        religious_count = sum(1 for term in self.religious_terms if term in text_lower or term in combined)

        total_bias = gender_count + political_count + religious_count

        return {
            "gender_bias": gender_count,
            "political_bias": political_count,
            "religious_bias": religious_count,
            "total_bias": total_bias
        }
