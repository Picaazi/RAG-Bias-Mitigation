Bias_groups = {
    "Race": ["Black", "White", "Asian", "Latino"],
    "Gender": ["Women", "Men", "Female", "Male"],
    "Politics": ["Conservative", "Democrat", "Republican"],
    "Religion": ["Christian", "Muslim", "Jewish", "Hindu"]
}
def count_groupmentions(text, group_terms):
    text_lower = str(text).lower()
    return sum(1 for term in group_terms if term.lower() in text_lower)
def biasamplicationscore(retrieved_docs, generated_response):
    total_response_G = sum(
        count_groupmentions(generated_response, terms)
        for terms in Bias_groups.values()
    )
    total_retrieved_G = sum(
        count_groupmentions(doc, terms)
        for doc in retrieved_docs
        for terms in Bias_groups.values()
    )

    biasscore = {}
    for g, terms in Bias_groups.items():
        retrieved_g = sum(count_groupmentions(doc, terms) for doc in retrieved_docs)
        retrieved_prop = retrieved_g / total_retrieved_G if total_retrieved_G > 0 else 0
        response_g = count_groupmentions(generated_response, terms)
        response_prop = response_g / total_response_G if total_response_G > 0 else 0

        biasscore[g] = response_prop - retrieved_prop

    return biasscore
