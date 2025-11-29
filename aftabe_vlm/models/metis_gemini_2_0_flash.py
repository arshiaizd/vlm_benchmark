from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import requests

from aftabe_vlm.models.base import VisionLanguageModel, ModelResponse


@dataclass
class MetisGemini20FlashConfig:
    api_key: Optional[str] = "tpsg-MNvTQUAqUL84o4THLV1395IqTBIZHJJ"
    # 👇 adjust to whatever Metis expects as the Gemini model id
    model_name: str = "gemini-2.5-flash"
    base_url: str = "https://api.metisai.ir"
    # 👇 provider name depends on how your Metis wrapper is configured;
    # if they expose Gemini through another provider, change this.
    provider: str = "google"
    max_tokens: int = 500
    timeout: int = 120


class MetisGemini20Flash(VisionLanguageModel):
    """
    Metis wrapper for a Gemini 2.0 Flash **vision** model.

    Assumes Metis exposes a chat/completions-style wrapper similar to
    the existing MetisGPT4o class, and that you select Gemini by the
    'model' name in the payload.

    Authentication:
      - Reads API key from:
          1) MetisGemini20FlashConfig.api_key, or
          2) METIS_API_KEY environment variable.
    """

    def __init__(self, config: Optional[MetisGemini20FlashConfig] = None) -> None:
        self.config = config or MetisGemini20FlashConfig()

        api_key = self.config.api_key or os.environ.get("METIS_API_KEY")
        if not api_key:
            raise RuntimeError(
                "MetisGemini20Flash requires a Metis API key. "
                "Set METIS_API_KEY env var or pass MetisGemini20FlashConfig(api_key=...)."
            )

        self.api_key = api_key
        self.model_name = self.config.model_name
        self.base_url = self.config.base_url
        self.provider = self.config.provider
        self.max_tokens = self.config.max_tokens
        self.timeout = self.config.timeout

        self.endpoint = (
            f"{self.base_url}/api/v1/wrapper/{self.provider}/chat/completions"
        )
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    @property
    def name(self) -> str:
        return f"metis-gemini-2.0-flash({self.model_name})"

    def _encode_image_as_data_url(self, image_path: str | Path) -> str:
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"Image file not found: {path}")
        data = path.read_bytes()
        b64 = base64.b64encode(data).decode("ascii")
        # Metis’ GPT-4o wrapper assumed PNGs; if yours are JPG, this still works
        return f"data:image/png;base64,{b64}"

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        image_path: str | Path,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> ModelResponse:
        image_data_url = self._encode_image_as_data_url(image_path)

        # Same OpenAI-style payload shape as MetisGPT4o
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_data_url,
                        },
                    },
                ],
            },
        ]

        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": self.max_tokens,
        }

        resp = requests.post(
            self.endpoint,
            json=payload,
            headers=self.headers,
            timeout=self.timeout,
        )
        if resp.status_code != 200:
            raise RuntimeError(
                f"MetisGemini20Flash API error {resp.status_code}: {resp.text}"
            )

        data = resp.json()

        # OpenAI-style response: choices[0].message.content
        message = (
            data.get("choices", [{}])[0]
            .get("message", {})
        )
        content = message.get("content", "")

        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            # content may be a list of parts (each with "type"/"text")
            parts_text = []
            for part in content:
                if isinstance(part, dict) and "text" in part:
                    parts_text.append(part["text"])
            text = "".join(parts_text)
        else:
            text = str(content)

        provider_payload: Dict[str, Any] = {
            "endpoint": self.endpoint,
            "model": self.model_name,
            "raw_response": data,
            "extra_metadata": extra_metadata,
        }

        return ModelResponse(raw_text=text, provider_payload=provider_payload)
