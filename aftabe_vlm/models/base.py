from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class ModelResponse:
    """Container for a single model call."""
    raw_text: str
    provider_payload: Optional[Dict[str, Any]] = None


class VisionLanguageModel(ABC):
    """Abstract base class for a VLM accessed via API."""

    name: str

    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        image_path: str,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> ModelResponse:
        """Call the underlying model and return its response."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
