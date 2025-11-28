from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Optional
import base64
import os
import requests

from .base import VisionLanguageModel, ModelResponse


class MetisGPT4o(VisionLanguageModel):
    """
    VisionLanguageModel implementation that talks to the Metis wrapper API.

    The API is treated as OpenAI-compatible chat/completions with vision support.

    Environment variable:
        METIS_API_KEY: the bearer token to use.
    """

    def __init__(
        self,
        api_key: Optional[str] = "tpsg-MNvTQUAqUL84o4THLV1395IqTBIZHJJ",
        model: str = "gpt-4o-mini-2024-07-18",
        base_url: str = "https://api.metisai.ir",
        provider: str = "openai_chat_completion",
        max_tokens: int = 500,
        timeout: int = 120,
    ):
        if api_key is None:
            api_key = os.environ.get("METIS_API_KEY")
        if not api_key:
            raise RuntimeError("MetisGPT4o requires an API key (set METIS_API_KEY env var or pass api_key).")

        self.api_key = api_key
        self.model_name = model
        self.base_url = base_url.rstrip("/")
        self.provider = provider
        self.max_tokens = max_tokens
        self.timeout = timeout

        self.endpoint = f"{self.base_url}/api/v1/wrapper/{self.provider}/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        self.name = f"metis-{self.model_name}"

    def _encode_image_as_data_url(self, image_path: str) -> str:
        img_bytes = Path(image_path).read_bytes()
        b64 = base64.b64encode(img_bytes).decode("utf-8")
        # MIME type is not strictly required; PNG is safe default
        return f"data:image/png;base64,{b64}"

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        image_path: str,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> ModelResponse:
        """Call the Metis API once and return the assistant's text output."""
        image_data_url = self._encode_image_as_data_url(image_path)

        # OpenAI-style messages, including image in content
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
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
            raise RuntimeError(f"Metis API error {resp.status_code}: {resp.text}")

        data = resp.json()
        choice = data["choices"][0]
        content = choice["message"]["content"]

        # If content is a simple string, use it; if it's parts, join text parts.
        if isinstance(content, str):
            raw_text = content
        else:
            fragments = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    fragments.append(part.get("text", ""))
            raw_text = "\n".join(fragments).strip()

        provider_payload: Dict[str, Any] = {
            "endpoint": self.endpoint,
            "model": self.model_name,
            "raw_response": data,
            "extra_metadata": extra_metadata or {},
        }

        return ModelResponse(raw_text=raw_text, provider_payload=provider_payload)
