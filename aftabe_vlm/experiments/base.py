from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from dataset import PuzzleSample


@dataclass
class ExperimentConfig:
    name: str
    description: str
    max_attempts: int = 1


class Experiment(ABC):
    """Base class for all experiments."""

    def __init__(self, name: str, description: str, max_attempts: int = 1):
        self.config = ExperimentConfig(name=name, description=description, max_attempts=max_attempts)

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def description(self) -> str:
        return self.config.description

    @property
    def max_attempts(self) -> int:
        return self.config.max_attempts

    @abstractmethod
    def build_hint_text(
        self,
        sample: PuzzleSample,
        attempt_index: int,
        previous_attempts: List[Dict[str, Any]],
    ) -> Optional[str]:
        """Build a human-readable hint string (or None) for this attempt."""
        raise NotImplementedError
