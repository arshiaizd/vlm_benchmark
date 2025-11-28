from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base import Experiment
from dataset import PuzzleSample


class RetryWithFeedbackExperiment(Experiment):
    """Experiment IV: retries with feedback on previous wrong guesses."""

    def __init__(self, max_attempts: int = 3):
        super().__init__(
            name="retry_with_feedback",
            description=(
                "Retry with feedback: on incorrect answers, re-prompt the model with its "
                "previous wrong guesses as negative hints."
            ),
            max_attempts=max_attempts,
        )

    def build_hint_text(
        self,
        sample: PuzzleSample,
        attempt_index: int,
        previous_attempts: List[Dict[str, Any]],
    ) -> Optional[str]:
        if attempt_index == 0 or not previous_attempts:
            return None

        guesses: List[str] = []
        for att in previous_attempts:
            parsed = att.get("parsed") or {}
            fa = parsed.get("final_answer")
            if fa:
                guesses.append(str(fa))

        if not guesses:
            return (
                "Your previous attempt(s) did not produce a usable final answer. "
                "Try again and be sure to output a single clear final_answer."
            )

        unique_guesses = sorted(set(guesses))
        guesses_str = ", ".join(f"'{g}'" for g in unique_guesses)

        return (
            "You previously attempted some answers, but ALL of them were wrong.\n"
            f"Your previous guesses were: {guesses_str}.\n"
            "Do NOT repeat these guesses. Think again carefully and propose a new answer."
        )
