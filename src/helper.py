import re

def extract_match_score(response_text):
    # Search for a number between 0 and 100
    match = re.search(r'\b(100|[1-9]?[0-9])\b', response_text)
    if match:
        return int(match.group(0))
    return 0