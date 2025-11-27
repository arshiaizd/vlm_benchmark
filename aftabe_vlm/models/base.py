from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class ModelResponse:
    """
    Container for a single model call.

    raw_text: full textual response produced by the model.
    provider_payload: optional nested structure with provider-specific metadata,
                     e.g., usage statistics, raw response dict, etc.
    """
    raw_text: str
    provider_payload: Optional[Dict[str, Any]] = None


class VisionLanguageModel(ABC):
    """
    Abstract base class for a VLM accessed via API.

    To add a new model (e.g., another provider), subclass this and
    implement `generate()`.
    """

    name: str

    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        image_path: str,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> ModelResponse:
        """
        Call the underlying model.

        Args:
            system_prompt: shared system-level instructions (game description).
            user_prompt: per-sample text (puzzle instructions + hints).
            image_path: path to the image file for this puzzle.
            extra_metadata: experiment-related metadata you might want to log.

        Returns:
            ModelResponse with raw_text and optional provider_payload.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
