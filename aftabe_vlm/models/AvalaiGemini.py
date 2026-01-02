from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Optional, List
import base64
import os
import requests

from .base import VisionLanguageModel, ModelResponse

class AvalAiGemini(VisionLanguageModel):
    """
    Gemini implementation that talks to the AvalAI (Metis) wrapper API.
    Since AvalAI wraps models in an OpenAI-compatible format, this class 
    uses the Chat Completions API structure but defaults to Gemini models.
    """

    def __init__(
        self,
        api_key: Optional[str] = "aa-roEXspUP5ipchJI5JdcDeew7WquYsRMSMJjkwCzRR1QCmoxz", 
        model: str = "gemini-2.5-flash", # Default to a Gemini model supported by AvalAI
        base_url: str = "https://api.avalapis.ir/v1",
        timeout: int = 480,
        temperature: Optional[float] = None,
    ):
        if api_key is None:
            api_key = os.environ.get("METIS_API_KEY") or "YOUR_FALLBACK_KEY"
            
        if not api_key:
            raise RuntimeError("AvalAiGemini requires an API key.")

        self.api_key = api_key
        self.model_name = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.temperature = temperature

        # AvalAI uses the standard OpenAI chat completions endpoint
        if self.base_url.endswith("/v1"):
            self.endpoint = f"{self.base_url}/chat/completions"
        else:
            self.endpoint = f"{self.base_url}/v1/chat/completions"

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        self.name = f"avalai-{self.model_name}"

    def _encode_image_as_data_url(self, image_path: str) -> str:
        """Encodes image to base64 data URL for OpenAI-compatible APIs."""
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
        """Single-turn generation wrapper."""
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
        Converts internal message format to OpenAI-compatible format used by AvalAI.
        """
        openai_messages = []
        
        for msg in messages:
            role = msg["role"]
            content_parts = []
            
            # Map 'model' role to 'assistant'
            if role == "model":
                role = "assistant"
            
            # Add text content
            if msg.get("text"):
                content_parts.append({"type": "text", "text": msg["text"]})
            
            # Add image content (AvalAI uses OpenAI's image_url structure)
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

        # Build Payload
        payload = {
            "model": self.model_name,
            "messages": openai_messages,
            # "stream": False
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
                raise RuntimeError(f"AvalAI Gemini Error {resp.status_code}: {resp.text}")
                
            data = resp.json()
            choice = data["choices"][0]
            content = choice["message"]["content"]
            
            if isinstance(content, str):
                raw_text = content
            else:
                raw_text = "\n".join([
                    part.get("text", "") 
                    for part in content 
                    if part.get("type") == "text"
                ])

        except Exception as e:
             raise RuntimeError(f"AvalAI Request Failed: {e}")

        provider_payload = {
            "endpoint": self.endpoint,
            "model": self.model_name,
            "raw_response": data,
            "extra_metadata": extra_metadata or {},
        }

        return ModelResponse(raw_text=raw_text, provider_payload=provider_payload)