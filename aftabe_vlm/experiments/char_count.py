from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base import Experiment
from ..dataset import PuzzleSample


class CharCountExperiment(Experiment):
    """Experiment II: provide the character count of the answer as a hint."""

    def __init__(self):
        super().__init__(
            name="char_count",
            description="Hint: character count of the answer (excluding spaces).",
            max_attempts=1,
        )

    def build_hint_text(
        self,
        sample: PuzzleSample,
        attempt_index: int,
        previous_attempts: List[Dict[str, Any]],
    ) -> Optional[str]:
        answer = sample.answer
        non_space_chars = [c for c in answer if not c.isspace()]
        n = len(non_space_chars)
        total_len = len(answer)

        return (
            f"The target answer has {n} non-space characters "
            f"({total_len} characters including spaces). Use this as a constraint."
        )
