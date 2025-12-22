# from typing import List, Dict

# # =====================================================
# # 1) SYSTEM PROMPTS (base list, shared by default)
# # =====================================================




# def get_base_prompts(category: str) -> str:
#     """
#     Return system prompts for a given dataset category, e.g. 'en', 'pe', 'cross'.

#     Right now all categories share SYSTEM_PROMPTS. If you want different ones
#     per category, branch on `category` here and/or define separate lists.
#     """

#     rules = None

#     if category == "en":
#         rules = "- The target answer language is English.\n\n"

#     elif category == "pe" or category == "fa":
#         rules = """
#         - The target answer language is Persian (Farsi).
#         - CULTURAL LENS: Do not simply translate English concepts. You must interpret the visual elements through the lens of Persian culture, literature, common daily idioms, etc.
#         - PUNS: If the image suggests a pun, look for phonetic similarities in Persian words, not English words.\n\n"""

#     elif category == "cross":
#         rules = """
#         - The target answer language is Persian (Farsi).
#         - ENGLISH KNOWLEDGE REQUIRED: The puzzle may rely on English words, concepts, or numbers depicted in the image.
#         - You may need to take these English elements and use them directly in the Persian answer (transliteration) or combine them to form the final Persian phrase.\n\n"""

#     else:
#         raise ValueError(f"Unknown category '{category}' for base prompts.")
    
#     first_part = """
#     You are an expert multi-modal puzzle solver. You solve picture word puzzles.

#     GAME DESCRIPTION:
#     - You will see exactly ONE image per puzzle.
#     - The image may depict objects, people, scenes, text, icons, or abstract compositions.
#     - The goal is to infer a SINGLE word or short phrase that the puzzle creator intends.
#     - The answer may be:
#     - a literal word matching objects in the image,
#     - an idiom or proverb,
#     - a pun,
#     - a common expression,
#     - or a culturally meaningful phrase.

#     LANGUAGE RULES:
#     """

#     last_part = """YOUR TASK:
#         Your task is to analyze the picture very carefully and identify the key words, symbols, or concepts it contains.
#         Then infer a single word or short phrase that meaningfully combines or represents them.
#         Take a deep breath, and think step by step until you arrive at your best answer.
#     """

#     return first_part + "\n\n" + rules + last_part




# # =====================================================
# # 2) USER PROMPTS (category-aware via function)
# # =====================================================

# def get_prompt_variants() -> List[Dict[str, str]]:
#     """
#     Return user prompts for a given answer language and dataset category.

#     For now, all categories share the same templates. You can customize
#     per-category by branching on `category`.
#     """

#     return [
#         {
#             "name": "user_v1",
#             "template": (
#                 """
#                     OUTPUT FORMAT:
#                         Return ONLY the final answer and DO NOT write out your reasoning steps. Output the final answer in the following format:
#                         "answer":[the inferred word or phrase]
#                         for example if the final answer is "cat", respond with:
#                         "answer":"cat"
#                 """
#             ),
#         },
#         {
#             "name": "user_v2",
#             "template": (
#                 """
#                     OUTPUT FORMAT:
#                         Provide a step-by-step analysis of the visual elements and linguistic connections. At the very end of your response, output the final answer in the following format:
#                         "answer":[the inferred word or phrase]
#                         for example if the final answer is "cat", respond with:
#                         "answer":"cat"
#                 """
#             ),
#         },
#         {
#             "name": "user_v3",
#             "template": (
#                 """
#                     OUTPUT FORMAT:
#                         Please structure your response with the following headers:
#                         1. VISUAL INVENTORY: List the specific objects, colors, or actions visible.
#                         2. SEMANTIC ASSOCIATIONS: List related idioms, puns, or synonyms based on the inventory.
#                         Provide a step-by-step analysis of the visual elements and linguistic connections.
#                         At the very end of your response, output the final answer in the following format:
#                         "answer":[the inferred word or phrase]
#                         for example if the final answer is "cat", respond with:
#                         "answer":"cat"
#                 """
#             ),
#         },
#     ]
#     # return [
#     #     {
#     #         "name": "user_v1",
#     #         "template": (
#     #             """
#     #                 OUTPUT FORMAT:
#     #                     Return ONLY a single valid JSON object. Do not write out your reasoning, markdown blocks or conversational text. Use the following keys:
#     #                     {
#     #                     "final_answer": "The inferred word or phrase"
#     #                     }
#     #             """
#     #         ),
#     #     },
#     #     {
#     #         "name": "user_v2",
#     #         "template": (
#     #             """
#     #                 OUTPUT FORMAT:
#     #                     Return ONLY a single valid JSON object. Do not output markdown blocks or conversational text. Use the following keys:
#     #                     {
#     #                     "reasoning": "Your step-by-step analysis of the visual elements and linguistic connections",
#     #                     "final_answer": "The inferred word or phrase"
#     #                     }
#     #             """
#     #         ),
#     #     },
#     #     {
#     #         "name": "user_v3",
#     #         "template": (
#     #             """
#     #                 OUTPUT FORMAT:
#     #                     Please structure your response with the following headers:
#     #                     1. VISUAL INVENTORY: List the specific objects, colors, or actions visible.
#     #                     2. SEMANTIC ASSOCIATIONS: List related idioms, puns, or synonyms based on the inventory.
#     #                     3. FINAL ANSWER: the single best solution.
#     #                     Return ONLY a single valid JSON object. Do not output markdown blocks or conversational text. Use the following keys:
#     #                     {
#     #                     "reasoning": "Your step-by-step analysis of the visual elements and linguistic connections",
#     #                     "final_answer": "The inferred word or phrase"
#     #                     }
#     #             """
#     #         ),
#     #     },
#     # ]


from typing import List, Dict

# =====================================================
# 1) SYSTEM PROMPTS (base list, shared by default)
# =====================================================

def get_base_prompts(category: str) -> str:
    """
    Return system prompts for a given dataset category, e.g. 'en', 'pe', 'cross'.

    Right now all categories share the same base prompt body plus category-specific
    language rules. Customize per category by branching on `category`.
    """

    rules = None

    if category == "en":
        rules = "- The target answer language is English.\n\n"

    elif category == "pe" or category == "fa":
        rules = """
- The target answer language is Persian (Farsi).
- CULTURAL LENS: Do not simply translate English concepts. You must interpret the visual elements through the lens of Persian culture, literature, and common daily idioms.
- WORDPLAY: If the image suggests wordplay, prioritize phonetic/semantic connections natural in Persian.\n\n"""

    elif category == "cross":
        rules = """
- The target answer language is Persian (Farsi).
- ENGLISH KNOWLEDGE REQUIRED: The puzzle may rely on English words, concepts, letters, or numbers depicted in the image.
- You may need to use English elements directly in the Persian answer (transliteration) or combine them with Persian to form the intended phrase.\n\n"""

    else:
        raise ValueError(f"Unknown category '{category}' for base prompts.")

    first_part = """
You are an expert multi-modal puzzle solver. You solve picture word puzzles.

GAME DESCRIPTION:
- You will see exactly ONE image per puzzle.
- The image may depict objects, people, scenes, text, icons, or abstract compositions.
- The goal is to infer a SINGLE intended answer: one word or a short phrase.
- The image is a deliberately constructed clue for a linguistic target, NOT a request to describe the scene.
- The intended answer may be:
  - a literal word,
  - an idiom or proverb,
  - a pun or wordplay,
  - a common expression,
  - a culturally meaningful phrase,
  - or a proper noun / named entity (person, place, title, brand, named item).

LANGUAGE RULES:
"""

    last_part = """
GENERAL SOLVING PROCEDURE (follow in order):
1) Identify candidate clue units in the image:
   - the most salient objects/entities
   - any text, letters, numbers, symbols, or icons
   - any repeated motif/pattern
2) Select ONLY 2–4 PRIMARY clue units:
   - prefer central/emphasized/repeated units
   - compress repeated motifs into one unit
   - ignore minor background details unless they clearly change a primary unit
3) Hypothesize a simple composition:
   - the answer is usually formed by combining or transforming the primary units
   - prefer the simplest coherent interpretation with the fewest assumptions
4) Choose the best final answer:
   - it should be natural/common in the target language
   - it should explain the primary units as a single intended construction
   - prioritize global coherence over matching every local detail

OUTPUT REQUIREMENT:
- Provide exactly ONE final answer (single word or short phrase).
- If uncertain, choose the most plausible candidate under the simplest coherent interpretation.
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
Return ONLY a single valid JSON object. Do not output markdown blocks or conversational text.
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
Return ONLY a single valid JSON object. Do not output markdown blocks or conversational text.
{
  "reasoning": "A concise explanation focusing on primary clue units and how they combine (avoid listing every minor detail).",
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
Return ONLY a single valid JSON object. Do not output markdown blocks or conversational text.
{
  "visual_inventory": ["List 2–4 PRIMARY clue units only (not every object)."],
  "final_answer": "The inferred word or phrase"
}
"""
            ),
        },
        {
            "name": "user_v4_controlled",
            "template": (
                """
OUTPUT FORMAT:
Return ONLY a single valid JSON object. Do not output markdown blocks or conversational text.
{
  "primary_clues": ["...", "..."],
  "candidates": ["...", "...", "..."],
  "final_answer": "..."
}

Rules:
- primary_clues must contain 2–4 items only.
- candidates must contain 3–5 items only.
- final_answer must be one of the candidates.
- Focus on global coherence; do not enumerate background details.
"""
            ),
        },
    ]

