from typing import List, Dict

# =====================================================
# 1) SYSTEM PROMPTS (base list, shared by default)
# =====================================================




def get_base_prompts(category: str) -> str:
    """
    Return system prompts for a given dataset category, e.g. 'en', 'pe', 'cross'.

    Right now all categories share SYSTEM_PROMPTS. If you want different ones
    per category, branch on `category` here and/or define separate lists.
    """

    rules = None

    if category == "en":
        rules = "- The target answer language is English.\n\n"

    elif category == "pe" or category == "fa":
        rules = """
        - The target answer language is Persian (Farsi).
        - CULTURAL LENS: Do not simply translate English concepts. You must interpret the visual elements through the lens of Persian culture, literature, common daily idioms, etc.
        - PUNS: If the image suggests a pun, look for phonetic similarities in Persian words, not English words.\n\n"""

    elif category == "cross":
        rules = """
        - The target answer language is Persian (Farsi).
        - ENGLISH KNOWLEDGE REQUIRED: The puzzle may rely on English words, concepts, or numbers depicted in the image.
        - You may need to take these English elements and use them directly in the Persian answer (transliteration) or combine them to form the final Persian phrase.\n\n"""

    else:
        raise ValueError(f"Unknown category '{category}' for base prompts.")
    
    first_part = """
    You are an expert multi-modal puzzle solver. You solve picture word puzzles.

    GAME DESCRIPTION:
    - You will see exactly ONE image per puzzle.
    - The image may depict objects, people, scenes, text, icons, or abstract compositions.
    - The goal is to infer a SINGLE word or short phrase that the puzzle creator intends.
    - The answer may be:
    - a literal word matching objects in the image,
    - an idiom or proverb,
    - a pun,
    - a common expression,
    - or a culturally meaningful phrase.

    LANGUAGE RULES:
    """

    last_part = """YOUR TASK:
        Your task is to analyze the picture very carefully and identify the key words, symbols, or concepts it contains.
        Then infer a single word or short phrase that meaningfully combines or represents them.
        Take a deep breath, and think step by step until you arrive at your best answer.
    """

    return first_part + "\n\n" + rules + last_part




# =====================================================
# 2) USER PROMPTS (category-aware via function)
# =====================================================

def get_prompt_variants(category: str) -> List[Dict[str, str]]:
    """
    Return user prompts for a given answer language and dataset category.

    For now, all categories share the same templates. You can customize
    per-category by branching on `category`.
    """

    return [
        {
            "name": "user_v1",
            "template": (
                """
                    OUTPUT FORMAT:
                        Return ONLY a single valid JSON object. Do not write out your reasoning, markdown blocks or conversational text. Use the following keys:
                        {
                        "final_answer": "The inferred word or phrase"
                        }
                """
            ),
        },
        {
            "name": "user_v2",
            "template": (
                """
                    OUTPUT FORMAT:
                        Return ONLY a single valid JSON object. Do not output markdown blocks or conversational text. Use the following keys:
                        {
                        "reasoning": "Your step-by-step analysis of the visual elements and linguistic connections",
                        "final_answer": "The inferred word or phrase"
                        }
                """
            ),
        },
        {
            "name": "user_v3",
            "template": (
                """
                    OUTPUT FORMAT:
                        Please structure your response with the following headers:
                        1. VISUAL INVENTORY: List the specific objects, colors, or actions visible.
                        2. SEMANTIC ASSOCIATIONS: List related idioms, puns, or synonyms based on the inventory.
                        3. FINAL ANSWER: the single best solution.
                        Return ONLY a single valid JSON object. Do not output markdown blocks or conversational text. Use the following keys:
                        {
                        "reasoning": "Your step-by-step analysis of the visual elements and linguistic connections",
                        "final_answer": "The inferred word or phrase"
                        }
                """
            ),
        },
    ]
