import os
import base64
import mimetypes
import requests
import json
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from pathlib import Path

from aftabe_vlm.models.base import VisionLanguageModel, ModelResponse

@dataclass
class GoogleVertexConfig:
    # Put your Google Cloud API Key here (starts with AIza...)
    api_key: Optional[str] = "AQ.Ab8RN6JUameBc8_AaLAu4VQIRIKpiKQYbshldsDW0Mq5pqShgg" 
    model_name: str = "gemini-2.5-flash" 
    base_url: str = "https://aiplatform.googleapis.com"
    timeout: int = 480

class GoogleVertexGemini(VisionLanguageModel):
    def __init__(self, config: Optional[GoogleVertexConfig] = None) -> None:
        self.config = config or GoogleVertexConfig()

        api_key = self.config.api_key or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("Set GOOGLE_API_KEY or pass GoogleVertexConfig(api_key=...).")

        self.api_key = api_key
        self.model_name = self.config.model_name
        self.base_url = self.config.base_url.rstrip("/")
        self.timeout = self.config.timeout

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
        """
        Single-turn generation (legacy support).
        """
        mime, image_b64 = self._encode_image_b64(image_path) if image_path else (None, None)

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
        }

        return self._send_request(payload, extra_metadata)

    def generate_chat(self, messages: List[Dict[str, Any]], extra_metadata: Optional[Dict[str, Any]] = None) -> ModelResponse:
        """
        Multi-turn chat generation.
        
        Args:
            messages: List of dicts, e.g.:
                [
                    {"role": "user", "text": "...", "image_path": "..."},
                    {"role": "model", "text": "..."}
                ]
        """
        contents = []
        
        for msg in messages:
            role = msg["role"]
            parts = []
            
            # Add text
            if msg.get("text"):
                parts.append({"text": msg["text"]})
            
            # Add image
            if msg.get("image_path"):
                mime, b64 = self._encode_image_b64(msg["image_path"])
                parts.append({"inline_data": {"mime_type": mime, "data": b64}})
                
            contents.append({"role": role, "parts": parts})

        payload = {
            "contents": contents,
            # "generationConfig": {
                # "temperature": 0.2,
                # "maxOutputTokens": 1000
            # }
        }
        
        return self._send_request(payload, extra_metadata)

    def _send_request(self, payload: Dict[str, Any], extra_metadata: Optional[Dict[str, Any]] = None) -> ModelResponse:
        """Helper to handle the actual HTTP request and parsing."""
        try:
            resp = requests.post(
                self.endpoint,
                params={"key": self.api_key},
                json=payload,
                headers=self.headers,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            raise RuntimeError(f"Google Vertex request failed: {e}") from e

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

