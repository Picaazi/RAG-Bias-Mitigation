import re

Bias_groups={
    "Race": ["Black", "White", "Asian", "Latino"],
    "Gender": ["Women", "Men", "Female", "Male","he","she"],
    "Politics": ["Conservative", "Democrat", "Republican","Left claim","Right "],
    "Religion": ["Christian", "Muslim", "Jewish", "Hindu"]
}

def extract_demographic_mentions(text):
    mentions = {category: [] for category in Bias_groups}
    for category, keywords in Bias_groups.items():
        for word in keywords:
            if re.search(rf"\b{word}\b", text, re.IGNORECASE):
                mentions[category].append(word)
    return mentions

# Example usage
response = "The elderly man and a young Muslim woman were at the event."
print(extract_demographic_mentions(response))


  
