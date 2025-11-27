from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import json
import re


@dataclass
class ParsedModelOutput:
    reasoning: Optional[str]
    final_answer: Optional[str]


def parse_model_response(raw_text: str) -> ParsedModelOutput:
    """
    Parse the model's response into (reasoning, final_answer).

    Primary expectation: the model returns a pure JSON object as instructed.
    We still handle mild deviations (extra text, etc.) defensively.
    """
    text = raw_text.strip()

    obj: Optional[Dict[str, Any]] = None

    # try direct JSON
    try:
        candidate = json.loads(text)
        if isinstance(candidate, dict):
            obj = candidate
    except Exception:
        obj = None

    # try to extract a JSON object substring
    if obj is None:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            json_str = match.group(0)
            try:
                candidate = json.loads(json_str)
                if isinstance(candidate, dict):
                    obj = candidate
            except Exception:
                obj = None

    reasoning: Optional[str] = None
    final_answer: Optional[str] = None

    if obj is not None:
        reasoning = obj.get("reasoning") or obj.get("thoughts") or obj.get("explanation")
        final_answer = obj.get("final_answer") or obj.get("answer")

    # fallback: if JSON parsing failed, try a crude pattern
    if final_answer is None:
        m = re.search(
            r"final_answer\s*[:=-]\s*['\"“”]?(.+?)['\"“”]?(?:$|\n)",
            text,
            flags=re.IGNORECASE,
        )
        if m:
            final_answer = m.group(1).strip()

    if reasoning is None:
        reasoning = text

    return ParsedModelOutput(reasoning=reasoning, final_answer=final_answer)


def normalize_answer(ans: str, language: Optional[str] = None) -> str:
    """
    Light normalization for comparing answers.

    - Trim whitespace
    - Collapse internal spaces
    - Lowercase
    - Basic Persian normalization: remove diacritics, unify Arabic/Farsi letters.
    """
    s = ans.strip()
    s = re.sub(r"\s+", " ", s)
    s = s.lower()

    if language:
        lang = language.lower()
    else:
        lang = ""

    if lang.startswith("fa") or "persian" in lang or "farsi" in lang:
        # Remove diacritics and tatweel
        s = re.sub(r"[\u064b-\u065f\u0670\u06d6-\u06ed\u0640]", "", s)
        # Normalize Arabic- vs Farsi- forms of yeh and kaf
        s = s.replace("\u064a", "\u06cc")  # ARABIC YEH -> FARSI YEH
        s = s.replace("\u0643", "\u06a9")  # ARABIC KAF -> FARSI KAF

    return s


def is_correct(
    predicted: Optional[str],
    gold: str,
    language: Optional[str] = None,
) -> bool:
    if not predicted:
        return False
    return normalize_answer(predicted, language) == normalize_answer(gold, language)


@dataclass
class SampleEvaluation:
    sample_id: str
    experiment_name: str
    model_name: str
    correct: bool
    attempts_used: int


def summarize_accuracy(records: List[SampleEvaluation]) -> Dict[str, Any]:
    """
    Simple aggregation utility for overall accuracy.
    """
    if not records:
        return {"accuracy": 0.0, "n": 0}

    n = len(records)
    correct = sum(1 for r in records if r.correct)
    return {
        "accuracy": correct / n,
        "n": n,
    }
