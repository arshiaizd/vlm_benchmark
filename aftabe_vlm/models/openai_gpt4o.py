from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
import base64

from openai import OpenAI

from .base import VisionLanguageModel, ModelResponse


class OpenAIGPT4o(VisionLanguageModel):
    """
    Concrete VLM implementation using OpenAI's GPT-4o (or similar) with vision.

    You must set the environment variable OPENAI_API_KEY before running.

    To add another OpenAI model variant, just instantiate with a different
    `model_name` or create a sibling class.
    """

    def __init__(self, model_name: str = "gpt-4o"):
        self.client = OpenAI()
        self.model_name = model_name
        self.name = model_name

    def _encode_image_as_data_url(self, image_path: str) -> str:
        img_bytes = Path(image_path).read_bytes()
        b64 = base64.b64encode(img_bytes).decode("utf-8")
        return f"data:image/png;base64,{b64}"

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        image_path: str,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> ModelResponse:
        image_data_url = self._encode_image_as_data_url(image_path)

        messages: list[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            },
        ]

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.2,
        )

        choice = response.choices[0]
        content = choice.message.content

        if isinstance(content, str):
            raw_text = content
        else:
            fragments = []
            for part in content:
                if getattr(part, "type", None) == "text":
                    fragments.append(part.text)
            raw_text = "\n".join(fragments).strip()

        provider_payload: Dict[str, Any] = {
            "model": response.model,
            "id": response.id,
            "usage": response.usage.model_dump() if hasattr(response.usage, "model_dump") else dict(response.usage),
            "extra_metadata": extra_metadata or {},
        }

        return ModelResponse(raw_text=raw_text, provider_payload=provider_payload)
