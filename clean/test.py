import string
import re

def normalize_answer(text: str) -> str:
    """
    Normalizes text to handle robustness cases:
    - Case insensitive
    - Ignores spaces ("match point" == "matchpoint")
    - Ignores hyphens ("match-point" == "matchpoint")
    - Ignores trailing punctuation
    """
    if not text:
        return ""
    
    # 1. Lowercase

    text = re.sub(r'[\u064B-\u0652]', '', text)
    
    text = text.lower()
    # 2. Remove punctuation (keeps letters/digits safe, removes .,-!?)
    # This map removes all standard punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # 3. Remove spaces
    text = text.replace(" ", "")
    
    return text

print(normalize_answer("دَبَّابَة"))