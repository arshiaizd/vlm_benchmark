from typing import List, Dict

# =====================================================
# 1) SYSTEM PROMPTS (base list, shared by default)
# =====================================================

def get_prompt(category: str) -> str:
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
    
    elif category == "ar":
        rules = """
                - The target answer language is Arabic.
                - CULTURAL LENS: Do not simply translate English concepts. You must interpret the visual elements through the lens of Arabic culture, literature, and common daily idioms.
                - WORDPLAY: If the image suggests wordplay, prioritize phonetic/semantic connections natural in Arabic.\n\n"""

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

    instruction = """
                OUTPUT FORMAT:
                Return ONLY a single valid JSON object. Do not output markdown blocks or conversational text.
                {
                "primary_clues": ["...", "..."],
                "candidates": ["...", "...", "..."],
                "final_answer": "..."
                }
                
                """
    return first_part + "\n\n" + rules + last_part + "\n\n" + instruction


