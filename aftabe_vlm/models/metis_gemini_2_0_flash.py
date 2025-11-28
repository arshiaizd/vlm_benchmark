from __future__ import annotations

import base64
import mimetypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import requests

from aftabe_vlm.models.base import VisionLanguageModel, ModelResponse


@dataclass
class MetisGeminiFlashConfig:
    api_key: Optional[str] = "tpsg-MNvTQUAqUL84o4THLV1395IqTBIZHJJ"
    model_name: str = "gemini-2.0-flash"
    base_url: str = "https://api.metisai.ir"
    max_output_tokens: int = 500
    timeout: int = 120


class MetisGemini20Flash(VisionLanguageModel):
    def __init__(self, config: Optional[MetisGeminiFlashConfig] = None) -> None:
        self.config = config or MetisGeminiFlashConfig()

        api_key = self.config.api_key or os.environ.get("METIS_API_KEY")
        if not api_key:
            raise RuntimeError("Set METIS_API_KEY or pass MetisGeminiFlashConfig(api_key=...).")

        self.api_key = api_key
        self.model_name = self.config.model_name
        self.base_url = self.config.base_url.rstrip("/")
        self.max_output_tokens = self.config.max_output_tokens
        self.timeout = self.config.timeout

        # Metis docs show Gemini via /v1beta/models/{model}:generateContent with x-goog-api-key
        self.endpoint = f"{self.base_url}/v1beta/models/{self.model_name}:generateContent"
        self.headers = {
            "x-goog-api-key": self.api_key,
            "Content-Type": "application/json",
        }

    @property
    def name(self) -> str:
        return f"metis-gemini({self.model_name})"

    def _encode_image_b64(self, image_path: str | Path) -> tuple[str, str]:
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"Image file not found: {path}")

        mime, _ = mimetypes.guess_type(str(path))
        if not mime or not mime.startswith("image/"):
            # safe fallback
            mime = "image/png"

        b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        return mime, b64

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        image_path: str | Path,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> ModelResponse:
        mime, image_b64 = self._encode_image_b64(image_path)

        # Gemini generateContent format (Metis docs show this API family)
        payload: Dict[str, Any] = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": f"{system_prompt}\n\n{user_prompt}".strip()},
                        {"inline_data": {"mime_type": mime, "data": image_b64}},
                    ],
                }
            ],
            "generationConfig": {"maxOutputTokens": self.max_output_tokens},
        }

        try:
            resp = requests.post(
                self.endpoint,
                json=payload,
                headers=self.headers,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            raise RuntimeError(f"Metis Gemini request failed: {e}") from e
        except ValueError as e:
            raise RuntimeError(f"Metis Gemini returned non-JSON: {resp.text[:500]}") from e

        # Gemini-style parsing
        text_parts: list[str] = []
        for cand in data.get("candidates", [])[:1]:
            for part in (cand.get("content", {}) or {}).get("parts", []) or []:
                if isinstance(part, dict) and "text" in part:
                    text_parts.append(part["text"])

        text = "".join(text_parts).strip()

        provider_payload: Dict[str, Any] = {
            "endpoint": self.endpoint,
            "model": self.model_name,
            "raw_response": data,
            "extra_metadata": extra_metadata,
        }
        return ModelResponse(raw_text=text, provider_payload=provider_payload)
