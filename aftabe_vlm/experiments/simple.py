from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base import Experiment
from dataset import PuzzleSample


class SimpleExperiment(Experiment):
    """Experiment I: no hints, just the image and base instructions."""

    def __init__(self):
        super().__init__(
            name="simple",
            description="Simple prompt: no hints.",
            max_attempts=1,
        )

    def build_hint_text(
        self,
        sample: PuzzleSample,
        attempt_index: int,
        previous_attempts: List[Dict[str, Any]],
    ) -> Optional[str]:
        return None
