from __future__ import annotations

from typing import Any, Dict, List, Optional
import math
import hashlib
import random

from .base import Experiment
from ..dataset import PuzzleSample


def _deterministic_indices(answer: str, sample_id: str, fraction: float = 0.1) -> List[int]:
    """Select a deterministic subset of indices for hint characters."""
    indices = [i for i, ch in enumerate(answer) if not ch.isspace()]
    if not indices:
        return []

    k = max(1, math.ceil(len(indices) * fraction))

    seed_bytes = hashlib.md5(sample_id.encode("utf-8")).digest()
    seed_int = int.from_bytes(seed_bytes, "big")
    rng = random.Random(seed_int)

    selected = indices.copy()
    rng.shuffle(selected)
    return sorted(selected[:k])


class PartialCharsExperiment(Experiment):
    """Experiment III: reveal 10% of characters with their positions."""

    def __init__(self):
        super().__init__(
            name="partial_chars",
            description="Hint: 10% of characters with their positions.",
            max_attempts=1,
        )

    def build_hint_text(
        self,
        sample: PuzzleSample,
        attempt_index: int,
        previous_attempts: List[Dict[str, Any]],
    ) -> Optional[str]:
        answer = sample.answer
        sample_id = sample.id

        indices = _deterministic_indices(answer, sample_id, fraction=0.10)
        if not indices:
            return "No character hints available (answer appears to be empty)."

        pattern_chars = []
        for i, ch in enumerate(answer):
            if ch.isspace():
                pattern_chars.append(" ")
            elif i in indices:
                pattern_chars.append(ch)
            else:
                pattern_chars.append("_")

        pattern = "".join(pattern_chars)

        pieces = []
        for i in indices:
            char = answer[i]
            pieces.append(f"position {i+1}: '{char}'")

        hints_list = "; ".join(pieces)
        return (
            "Some letters of the answer are revealed.\n"
            f"Pattern (underscores = unknown letters, spaces preserved): {pattern}\n"
            f"Known characters (1-based index): {hints_list}"
        )
