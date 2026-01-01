import base64
import mimetypes
import os
from pathlib import Path
from typing import Any, Dict, Optional, List

import requests

# Adjust import to match your project
from aftabe_vlm.models.base import VisionLanguageModel, ModelResponse


class LlamaVision(VisionLanguageModel):
    """
    VisionLanguageModel implementation for Meta Llama Vision via AvalAI.

    Common AvalAI model IDs:
      - "llama-3.2-11b-vision-instruct"
      - "meta.llama-3.2-90b-vision-instruct"
    """

    def __init__(
        self,
        api_key: Optional[str] = "aa-roEXspUP5ipchJI5JdcDeew7WquYsRMSMJjkwCzRR1QCmoxz",
        model: str = "llama-4-scout-17b-16e-instruct",
        base_url: str = "https://api.avalai.ir/v1",
        timeout: int = 120,
    ):
        # Prefer environment variable to avoid hardcoding secrets in code
        api_key = api_key or os.getenv("AVALAI_API_KEY")
        if not api_key:
            raise RuntimeError("AvalAI API key is required (pass api_key=... or set AVALAI_API_KEY).")

        self.api_key = api_key
        self.model_name = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

        # OpenAI-compatible Chat Completions endpoint
        self.endpoint = f"{self.base_url}/chat/completions" if self.base_url.endswith("/v1") else f"{self.base_url}/v1/chat/completions"

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    @property
    def name(self) -> str:
        return f"avalai-{self.model_name}"

    def _encode_image_as_data_url(self, image_path: str) -> str:
        # If it's already a URL, pass-through
        if image_path.startswith(("http://", "https://")):
            return image_path

        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")

        img_bytes = path.read_bytes()
        b64 = base64.b64encode(img_bytes).decode("utf-8")

        mime, _ = mimetypes.guess_type(str(path))
        mime = mime or "image/jpeg"  # fallback
        return f"data:{mime};base64,{b64}"

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        image_path: Optional[str] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
        max_tokens: Optional[int] = None,
    ) -> ModelResponse:
        """Single-turn wrapper (system + user, optional image)."""
        messages = [
            {"role": "system", "text": system_prompt},
            {"role": "user", "text": user_prompt, "image_path": image_path},
        ]
        return self.generate_chat(messages, extra_metadata=extra_metadata, max_tokens=max_tokens)

    def generate_chat(
        self,
        messages: List[Dict[str, Any]],
        extra_metadata: Optional[Dict[str, Any]] = None,
        max_tokens: Optional[int] = None,
    ) -> ModelResponse:
        """Multi-turn chat generation."""
        openai_messages = []

        for msg in messages:
            role = msg["role"]
            if role == "model":
                role = "assistant"

            # System messages are typically plain string content
            if role == "system":
                openai_messages.append({"role": "system", "content": msg.get("text", "")})
                continue

            content_parts = []

            if msg.get("text"):
                content_parts.append({"type": "text", "text": msg["text"]})

            if msg.get("image_path"):
                data_url = self._encode_image_as_data_url(msg["image_path"])
                content_parts.append({"type": "image_url", "image_url": {"url": data_url}})

            openai_messages.append({"role": role, "content": content_parts})

        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": openai_messages,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        try:
            resp = requests.post(
                self.endpoint,
                json=payload,
                headers=self.headers,
                timeout=self.timeout,
            )
            if resp.status_code != 200:
                raise RuntimeError(f"AvalAI Error {resp.status_code}: {resp.text}")

            data = resp.json()
            choice = data["choices"][0]
            content = choice["message"]["content"]

            # AvalAI/OpenAI-style responses sometimes return string or content-parts
            if isinstance(content, str):
                raw_text = content
            else:
                raw_text = "\n".join(
                    [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
                )

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
#     client = LlamaVision(
#     )

#     image_path = r"D:\study\agha omid\vlm_benchmark\dataset\en\en_images\1.jpg"

#     resp = client.generate(
#         system_prompt="You are a helpful vision assistant.",
#         user_prompt="Describe the image in 3 bullet points.",
#         image_path=image_path,
#         max_tokens=500,
#     )

#     print(resp.raw_text)


# if __name__ == "__main__":
#     main()
