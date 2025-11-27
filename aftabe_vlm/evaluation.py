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
    """Parse the model's response into (reasoning, final_answer)."""
    text = raw_text.strip()

    obj: Optional[Dict[str, Any]] = None

    try:
        candidate = json.loads(text)
        if isinstance(candidate, dict):
            obj = candidate
    except Exception:
        obj = None

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
    """Normalize answer for comparison (basic, with Persian support)."""
    s = ans.strip()
    s = re.sub(r"\s+", " ", s)
    s = s.lower()

    if language:
        lang = language.lower()
    else:
        lang = ""

    if lang.startswith("fa") or "persian" in lang or "farsi" in lang:
        s = re.sub(r"[\u064b-\u065f\u0670\u06d6-\u06ed\u0640]", "", s)
        s = s.replace("\u064a", "\u06cc")
        s = s.replace("\u0643", "\u06a9")

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
    """Simple aggregation utility for overall accuracy."""
    if not records:
        return {"accuracy": 0.0, "n": 0}

    n = len(records)
    correct = sum(1 for r in records if r.correct)
    return {
        "accuracy": correct / n,
        "n": n,
    }
