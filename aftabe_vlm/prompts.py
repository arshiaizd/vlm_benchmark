from __future__ import annotations

from typing import List, Dict
from typing import Optional
from .dataset import PuzzleSample

# BASE_SYSTEM_PROMPT = """
# You are an expert multi-modal puzzle solver. You solve picture word puzzles similar to
# the Iranian mobile game “Aftabe” and related English / Persian / cross-lingual variants.

# GAME DESCRIPTION (for you, the model):
# - You will see exactly ONE image per puzzle.
# - The image may depict objects, people, scenes, text, icons, or abstract compositions.
# - The goal is to infer a SINGLE word or short phrase that the puzzle creator intends.
# - The answer may be:
#   - a literal word matching objects in the image,
#   - an idiom or proverb,
#   - a pun,
#   - a common expression,
#   - or a culturally meaningful phrase.
# - The answer language can be English, Persian (Farsi), or a cross-lingual expression
#   mixing the two; this will be indicated in the instructions.

# YOUR TASK:
# - Carefully inspect the provided image.
# - Combine visual details with any hints (e.g., answer length, known characters, previous wrong guesses).
# - Infer the most likely intended target word or phrase, in the requested language.
# - Do NOT output multiple candidate answers. Choose exactly ONE final answer.

# RESPONSE FORMAT (VERY IMPORTANT):
# You MUST respond with ONLY a single JSON object, with exactly these two keys:
#   {{
#     "reasoning": "<your step-by-step reasoning in plain text>",
#     "final_answer": "<your SINGLE best guess word or phrase>"
#   }}

# Constraints:
# - Do NOT include any additional keys.
# - Do NOT wrap the JSON in backticks or any other text.
# - "final_answer" must be a short string: just the guessed word/phrase, no explanation.
# - "reasoning" can be several sentences explaining how you interpreted the image and hints.
# - Follow any language constraints given in the user message (e.g. “answer must be in Persian”).

# If you are uncertain, still choose your single best guess and explain your uncertainty in "reasoning".
# """
BASE_SYSTEM_PROMPT ="""
"""


def build_puzzle_user_prompt(
    sample: PuzzleSample,
    hint_text: Optional[str] = None,
    variant: Optional[Dict[str, str]] = None,  # <--- Type hint updated
) -> str:
    """Build the user message text for a puzzle, optionally with hints and variants."""
    base = get_base_prompts(sample.answer_language)
    
    # 1. Append the specific instruction for this variant (v1/v2/v3)
    if variant and "template" in variant:
        base += "\n" + variant["template"]

    # 2. Add hints if they exist
    if hint_text:
        base += "\n\n" + "HINT: " + hint_text

    return base



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

def get_prompt_variants() -> List[Dict[str, str]]:
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
                        Return ONLY the final answer and DO NOT write out your reasoning steps. Output the final answer in the following format:
                        "answer":[the inferred word or phrase]
                """
            ),
        },
        {
            "name": "user_v2",
            "template": (
                """
                    OUTPUT FORMAT:
                        Provide a step-by-step analysis of the visual elements and linguistic connections. At the very end of your response, output the final answer in the following format:
                        "answer":[the inferred word or phrase]
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
                        Provide a step-by-step analysis of the visual elements and linguistic connections.
                        At the very end of your response, output the final answer in the following format:
                        "answer":[the inferred word or phrase]
                """
            ),
        },
    ]
    # return [
    #     {
    #         "name": "user_v1",
    #         "template": (
    #             """
    #                 OUTPUT FORMAT:
    #                     Return ONLY a single valid JSON object. Do not write out your reasoning, markdown blocks or conversational text. Use the following keys:
    #                     {
    #                     "final_answer": "The inferred word or phrase"
    #                     }
    #             """
    #         ),
    #     },
    #     {
    #         "name": "user_v2",
    #         "template": (
    #             """
    #                 OUTPUT FORMAT:
    #                     Return ONLY a single valid JSON object. Do not output markdown blocks or conversational text. Use the following keys:
    #                     {
    #                     "reasoning": "Your step-by-step analysis of the visual elements and linguistic connections",
    #                     "final_answer": "The inferred word or phrase"
    #                     }
    #             """
    #         ),
    #     },
    #     {
    #         "name": "user_v3",
    #         "template": (
    #             """
    #                 OUTPUT FORMAT:
    #                     Please structure your response with the following headers:
    #                     1. VISUAL INVENTORY: List the specific objects, colors, or actions visible.
    #                     2. SEMANTIC ASSOCIATIONS: List related idioms, puns, or synonyms based on the inventory.
    #                     3. FINAL ANSWER: the single best solution.
    #                     Return ONLY a single valid JSON object. Do not output markdown blocks or conversational text. Use the following keys:
    #                     {
    #                     "reasoning": "Your step-by-step analysis of the visual elements and linguistic connections",
    #                     "final_answer": "The inferred word or phrase"
    #                     }
    #             """
    #         ),
    #     },
    # ]
