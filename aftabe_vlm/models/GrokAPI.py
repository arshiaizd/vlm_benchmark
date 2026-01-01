import base64
import mimetypes
from pathlib import Path
from typing import Any, Dict, Optional, List

import requests
from clean.prompt import get_prompt
# Adjust import to match your project
from aftabe_vlm.models.base import VisionLanguageModel, ModelResponse

class GrokAPI(VisionLanguageModel):
    """
    VisionLanguageModel implementation for xAI Grok (xAI API).

    Base:
      - https://api.x.ai
    Endpoint:
      - /v1/chat/completions  (OpenAI-compatible)  :contentReference[oaicite:2]{index=2}

    Common model ids (check your account availability):
      - "grok-4" (reasoning + vision) :contentReference[oaicite:3]{index=3}
      - "grok-4-fast-non-reasoning" (fast) :contentReference[oaicite:4]{index=4}
      - "grok-4-1-fast-non-reasoning" (agentic/tooling oriented) :contentReference[oaicite:5]{index=5}
    """

    def __init__(
        self,
        api_key: Optional[str] = "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",
        model: str = "x-ai/grok-4.1-fast",
        base_url: str = "https://openrouter.ai/api/v1",
        timeout: int = 120,
        image_detail: str = "high",  # "low" | "high" (if supported by the model)
    ):
        if not api_key:
            raise RuntimeError("API key is required (use env var XAI_API_KEY).")

        self.api_key = api_key
        self.model_name = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.image_detail = image_detail

        # If base_url already ends with /v1, append /chat/completions, else append /v1/chat/completions
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
        return f"xai-{self.model_name}"

    def _encode_image_as_data_url(self, image_path_or_url: str) -> str:
        # If it's already a URL, pass-through
        if image_path_or_url.startswith("http://") or image_path_or_url.startswith("https://"):
            return image_path_or_url

        path = Path(image_path_or_url)
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")

        img_bytes = path.read_bytes()
        b64 = base64.b64encode(img_bytes).decode("utf-8")

        mime, _ = mimetypes.guess_type(str(path))
        if not mime:
            mime = "image/jpeg"

        return f"data:{mime};base64,{b64}"

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        image_path: str,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> ModelResponse:
        """Single-turn wrapper."""
        messages = [
            {"role": "system", "text": system_prompt},
            {"role": "user", "text": user_prompt, "image_path": image_path},
        ]
        return self.generate_chat(messages, extra_metadata)

    def generate_chat(
        self,
        messages: List[Dict[str, Any]],
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> ModelResponse:
        openai_messages = []

        for msg in messages:
            role = msg["role"]
            if role == "model":
                role = "assistant"

            content_parts = []

            if msg.get("text"):
                content_parts.append({"type": "text", "text": msg["text"]})

            if msg.get("image_path"):
                data_url_or_url = self._encode_image_as_data_url(msg["image_path"])
                content_parts.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": data_url_or_url,
                            "detail": self.image_detail,
                        },
                    }
                )

            openai_messages.append({"role": role, "content": content_parts})

        payload = {
            "model": self.model_name,
            "messages": openai_messages,
        }

        try:
            resp = requests.post(
                self.endpoint,
                json=payload,
                headers=self.headers,
                timeout=self.timeout,
            )
            if resp.status_code != 200:
                raise RuntimeError(f"xAI Error {resp.status_code}: {resp.text}")

            data = resp.json()
            choice = data["choices"][0]
            content = choice["message"]["content"]

            # Usually a string, but keep fallback
            if isinstance(content, str):
                raw_text = content
            else:
                raw_text = "\n".join(
                    [p.get("text", "") for p in content if p.get("type") == "text"]
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


# def build_base_mode_user_text(lang: str) -> str:
#     """
#     EXACT base mode from your benchmark (USE_CONTEXT=False):
#       f"{sys_prompt}\\n\\nAnalyze the image and provide the JSON solution."
#     """
#     sys_prompt = get_prompt(lang)
#     return f"{sys_prompt}\n\nAnalyze the image and provide the JSON solution."


# def main():

#     client = GrokAPI()

#     image_path = r"D:\study\agha omid\vlm_benchmark\dataset\en\en_images\168.jpg"
#     lang = "en"

#     # Build prompt exactly like your benchmark base mode
#     user_text = build_base_mode_user_text(lang)

#     # Option A (closest to benchmark): call generate_chat with single user msg
#     resp = client.generate(system_prompt="", user_prompt=user_text, image_path=image_path)
#     print(resp.raw_text)



# if __name__ == "__main__":
#     main()