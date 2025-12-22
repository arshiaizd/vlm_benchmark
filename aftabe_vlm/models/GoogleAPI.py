import os
import base64
import mimetypes
import requests
from dataclasses import dataclass
from typing import Optional, Dict, Any
from pathlib import Path

import requests

from aftabe_vlm.models.base import VisionLanguageModel, ModelResponse

# Assuming you have these base classes defined in your project
# from your_module import VisionLanguageModel, ModelResponse 

@dataclass
class GoogleVertexConfig:
    # Put your Google Cloud API Key here (starts with AIza...)
    api_key: Optional[str] = "AQ.Ab8RN6JUameBc8_AaLAu4VQIRIKpiKQYbshldsDW0Mq5pqShgg" 
    # The model name from your curl command
    model_name: str = "gemini-2.5-flash" 
    # The Vertex AI global base URL
    base_url: str = "https://aiplatform.googleapis.com"
    # max_output_tokens: int = 500
    timeout: int = 120


class GoogleVertexGemini(VisionLanguageModel):
    def __init__(self, config: Optional[GoogleVertexConfig] = None) -> None:
        self.config = config or GoogleVertexConfig()

        # Check for GOOGLE_API_KEY in environment variables if not provided in config
        api_key = self.config.api_key or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("Set GOOGLE_API_KEY or pass GoogleVertexConfig(api_key=...).")

        self.api_key = api_key
        self.model_name = self.config.model_name
        self.base_url = self.config.base_url.rstrip("/")
        # self.max_output_tokens = self.config.max_output_tokens
        self.timeout = self.config.timeout

        # Construct the URL exactly as shown in your curl documentation
        # Note: We use :generateContent (blocking) instead of :streamGenerateContent
        self.endpoint = f"{self.base_url}/v1/publishers/google/models/{self.model_name}:generateContent"
        
        self.headers = {
            "Content-Type": "application/json",
        }

    @property
    def name(self) -> str:
        return f"google-vertex({self.model_name})"

    def _encode_image_b64(self, image_path: str | Path) -> tuple[str, str]:
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"Image file not found: {path}")

        mime, _ = mimetypes.guess_type(str(path))
        if not mime or not mime.startswith("image/"):
            # Safe fallback
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
        mime, image_b64 = self._encode_image_b64(image_path) if image_path else (None, None)

        # Gemini payload structure
        payload: Dict[str, Any] = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": f"{system_prompt}\n\n{user_prompt}".strip()},
                        {"inline_data": {"mime_type": mime, "data": image_b64}},
                    ] if mime is not None else [
                        {"text": f"{system_prompt}\n\n{user_prompt}".strip()}
                    ],
                }
            ],
            # "generationConfig": {
                # "maxOutputTokens": self.max_output_tokens
            # },
        }

        try:
            # CHANGE: Pass the API Key as a query parameter (?key=YOUR_KEY)
            resp = requests.post(
                self.endpoint,
                params={"key": self.api_key},
                json=payload,
                headers=self.headers,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            # If you get a 404, double check the 'model_name' or your API Key permissions
            raise RuntimeError(f"Google Vertex request failed: {e}") from e
        except ValueError as e:
            raise RuntimeError(f"Google Vertex returned non-JSON: {resp.text[:500]}") from e

        # Parse the response
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