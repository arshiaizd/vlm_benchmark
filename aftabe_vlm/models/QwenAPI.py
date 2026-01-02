import os
import base64
from pprint import pprint

import requests
import json
from pathlib import Path
from typing import Any, Dict, Optional, List

from openai import OpenAI

# Adjust import to match your project
from aftabe_vlm.models.base import VisionLanguageModel, ModelResponse


class Qwen3(VisionLanguageModel):
    """
    VisionLanguageModel implementation for Qwen 3 VL via AvalAI.

    MODELS (Dec 2025):
    - "qwen3-vl-plus": The current FLAGSHIP multimodal model. Best for complex reasoning/OCR.
    - "qwen3-vl-flash": Faster, cheaper, but slightly less capable.
    - "qwen-vl-max": Usually aliases to the best stable version (likely Qwen 3 VL now).
    """

    def __init__(
            self,
            api_key: Optional[str] = "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",
            model: str = "qwen/qwen3-vl-235b-a22b-instruct",  # Updated to Qwen 3 Flagship
            base_url: str = "https://openrouter.ai/api/v1",
            timeout: int = 120,
            temperature: Optional[float] = None,
    ):
        if not api_key:
            raise RuntimeError("API key is required.")

        self.api_key = api_key
        self.model_name = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.temperature = temperature

        if self.base_url.endswith("/v1"):
            self.endpoint = f"{self.base_url}/chat/completions"
        else:
            self.endpoint = f"{self.base_url}/v1/chat/completions"

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    @property
    def name(self) -> str:
        return f"avalai-{self.model_name}"

    def _encode_image_as_data_url(self, image_path: str) -> str:
        if image_path.startswith("http"):
            return image_path

        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")

        img_bytes = path.read_bytes()
        b64 = base64.b64encode(img_bytes).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"

    def generate(
            self,
            system_prompt: str,
            user_prompt: str,
            image_path: str,
            extra_metadata: Optional[Dict[str, Any]] = None,
            temperature: Optional[float] = None,
    ) -> ModelResponse:
        """Single-turn wrapper."""
        messages = [
            {"role": "user", "text": f"{system_prompt}\n\n{user_prompt}", "image_path": image_path}
        ]
        return self.generate_chat(messages, extra_metadata, temperature)

    def generate_chat(
            self,
            messages: List[Dict[str, Any]],
            extra_metadata: Optional[Dict[str, Any]] = None,
            temperature: Optional[float] = None
    ) -> ModelResponse:
        """
        Multi-turn chat generation.
        """
        openai_messages = []

        for msg in messages:
            role = msg["role"]
            if role == "model":
                role = "assistant"

            content_parts = []

            # Add text
            if msg.get("text"):
                content_parts.append({"type": "text", "text": msg["text"]})

            # Add image
            if msg.get("image_path"):
                data_url = self._encode_image_as_data_url(msg["image_path"])
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": data_url}
                })

            openai_messages.append({
                "role": role,
                "content": content_parts
            })

        payload = {
            "model": self.model_name,
            "messages": openai_messages,
            # Qwen 3 supports massive contexts, but safe default helps latency
            # "max_tokens": 4000
        }

        temp = self.temperature if temperature is None else temperature
        if temp is not None:
            payload["temperature"] = float(temp)

        try:
            resp = requests.post(
                self.endpoint,
                json=payload,
                headers=self.headers,
                timeout=self.timeout,
            )

            if resp.status_code != 200:
                raise RuntimeError(f"AvalAI Qwen 3 Error {resp.status_code}: {resp.text}")

            data = resp.json()
            choice = data["choices"][0]
            content = choice["message"]["content"]

            if isinstance(content, str):
                raw_text = content
            else:
                raw_text = "\n".join([p.get("text", "") for p in content if p.get("type") == "text"])

        except Exception as e:
            raise RuntimeError(f"Request Failed: {e}")

        provider_payload = {
            "endpoint": self.endpoint,
            "model": self.model_name,
            "raw_response": data,
            "extra_metadata": extra_metadata or {},
        }

        return ModelResponse(raw_text=raw_text, provider_payload=provider_payload)


# def main():
#     client = Qwen3(
#     )

#     image_path = "D:\\study\\agha omid\\vlm_benchmark\dataset\en\en_images\\1.jpg"

#     image_url = "https://picsum.photos/512" # random image
#     resp = client.generate(
#         system_prompt="You are a helpful vision assistant.",
#         user_prompt="describe the picture.",
#         image_path=image_path,
#     )
#     print("\n=== URL IMAGE RESULT ===")
#     print(resp.raw_text)


# api = "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"

# def x():
#     client = OpenAI(
#         base_url="https://openrouter.ai/api/v1",
#         api_key=api,
#     )

#     completion = client.chat.completions.create(
#         extra_body={},
#         model="qwen/qwen3-vl-235b-a22b-instruct",
#         messages=[
#             {
#                 "role": "user",
#                 "content": [
#                     {
#                         "type": "text",
#                         "text": "What is in this image?"
#                     },
#                     {
#                         "type": "image_url",
#                         "image_url": {
#                             "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
#                         }
#                     }
#                 ]
#             }
#         ]
#     )
#     print(completion.choices[0].message.content)

# if __name__ == "__main__":
#     main()