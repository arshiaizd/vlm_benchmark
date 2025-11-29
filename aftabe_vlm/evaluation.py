from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Dict, Any,Sequence
import unicodedata
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


PERSIAN_DIACRITICS_RE = re.compile(r"[\u064b-\u065f\u0670\u06d6-\u06ed\u0640]")


def normalize_answer(ans: str, language: Optional[str] = None) -> str:
    """
    Normalize answer for comparison (English + Persian).

    Steps:
      - Unicode normalize (NFKC)
      - strip outer whitespace
      - collapse internal whitespace
      - lowercase
      - Persian: remove diacritics, normalize Arabic yeh/kaf to Persian, remove ZWNJ
      - finally remove ALL whitespace and dashes so:
          "wall street" == "wall-street" == "WallStreet"
    """
    if ans is None:
        return ""

    # Unicode normalize
    s = unicodedata.normalize("NFKC", ans)

    # Trim + collapse whitespace
    s = s.strip()
    s = re.sub(r"\s+", " ", s)

    # Lowercase
    s = s.lower()

    lang = (language or "").lower()

    # Persian-specific normalization
    if (
        lang.startswith("fa")
        or "persian" in lang
        or "farsi" in lang
        or lang.startswith("pe")
    ):
        # Remove diacritics + tatweel
        s = PERSIAN_DIACRITICS_RE.sub("", s)
        # Unify Arabic yeh/kaf to Persian forms
        s = s.replace("\u064a", "\u06cc")  # ي -> ی
        s = s.replace("\u0643", "\u06a9")  # ك -> ک
        # Remove zero-width non-joiner
        s = s.replace("\u200c", "")

    # Remove all whitespace and dashes for final comparison
    # e.g. "wall street", "wall-street", "wallstreet" -> "wallstreet"
    s = re.sub(r"[\s\-]+", "", s)

    return s


def _levenshtein(a: str, b: str) -> int:
    """Classic Levenshtein distance (edit distance)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    # dp over rows
    prev_row = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur_row = [i]
        for j, cb in enumerate(b, start=1):
            insert_cost = cur_row[j - 1] + 1
            delete_cost = prev_row[j] + 1
            replace_cost = prev_row[j - 1] + (ca != cb)
            cur_row.append(min(insert_cost, delete_cost, replace_cost))
        prev_row = cur_row
    return prev_row[-1]


def _fuzzy_equal(a: str, b: str, max_edits: int) -> bool:
    """
    Check if two normalized strings are equal within max_edits Levenshtein distance.
    """
    if a == b:
        return True
    # quick length guard to avoid wasting time
    if abs(len(a) - len(b)) > max_edits:
        return False
    return _levenshtein(a, b) <= max_edits


def is_correct(
    predicted: Optional[str],
    gold: str,
    language: Optional[str] = None,
    alt_answers: Optional[Sequence[str]] = None,
    max_edit_distance: int = 0,
) -> bool:
    """
    Return True if predicted answer matches gold (or any alt answer) under
    normalization + typo tolerance.

    - normalization handles:
        * case, unicode, spaces, dashes
        * Persian diacritics, ی/ک, ZWNJ
    - typo tolerance is controlled by max_edit_distance (default: 1)
    - alt_answers: optional list of extra valid answers
    """
    if not predicted:
        return False



    n_pred = normalize_answer(predicted, language)
    if not n_pred:
        return False

    # ---- main gold ----
    n_gold = normalize_answer(gold, language)

    # 1) exact normalized equality
    if n_pred == n_gold:
        return True

    # 2) fuzzy equality (typo tolerance)
    if _fuzzy_equal(n_pred, n_gold, max_edit_distance):
        return True

    # ---- alternate gold answers (optional) ----
    if alt_answers:
        for alt in alt_answers:
            n_alt = normalize_answer(alt, language)
            if n_pred == n_alt or _fuzzy_equal(n_pred, n_alt, max_edit_distance):
                return True

    return False

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


if __name__=="__main__":
    s1 = input()
    s2 = input()
    ans = is_correct(s1 , s2 , "english" , max_edit_distance=0)
    print(ans)
