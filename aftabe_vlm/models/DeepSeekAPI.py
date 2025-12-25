from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Optional, List
import base64
import os
import requests

# Adjust these imports to match your project structure
from .base import VisionLanguageModel, ModelResponse


class DeepSeekAPI(VisionLanguageModel):
    """
    VisionLanguageModel implementation that talks to the AvalAI platform.
    AvalAI provides access to models like GPT-4, Claude, and DeepSeek via a unified API.
    """

    def __init__(
        self,
        api_key: Optional[str] = "aa-roEXspUP5ipchJI5JdcDeew7WquYsRMSMJjkwCzRR1QCmoxz",
        model: str = "deepseek-v3.2", # AvalAI supports 'deepseek-chat', 'gpt-4o', etc.
        base_url: str = "https://api.avalai.ir/v1", # Official AvalAI endpoint
        provider: str = "avalai",
        timeout: int = 120,
    ):
        if api_key is None:
            # Common env var for this platform
            api_key = os.environ.get("AVALAI_API_KEY")
            
        if not api_key:
            raise RuntimeError("AvalAI requires an API key. Please set AVALAI_API_KEY or pass it explicitly.")

        self.api_key = api_key
        self.model_name = model
        self.base_url = base_url.rstrip("/")
        self.provider = provider
        self.timeout = timeout

        # AvalAI uses standard OpenAI-compatible paths
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
    ) -> ModelResponse:
        """Single-turn wrapper"""
        messages = [
            {"role": "user", "text": f"{system_prompt}\n\n{user_prompt}", "image_path": image_path}
        ]
        return self.generate_chat(messages, extra_metadata)

    def generate_chat(
        self, 
        messages: List[Dict[str, Any]], 
        extra_metadata: Optional[Dict[str, Any]] = None
    ) -> ModelResponse:
        """
        Multi-turn chat generation compatible with main loop.
        """
        openai_messages = []
        
        for msg in messages:
            role = msg["role"]
            content_parts = []
            
            if role == "model":
                role = "assistant"
            
            # Add text
            if msg.get("text"):
                content_parts.append({"type": "text", "text": msg["text"]})
            
            # Add image (AvalAI supports OpenAI vision format for models like gpt-4-vision)
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
        }
        
        try:
            resp = requests.post(
                self.endpoint,
                json=payload,
                headers=self.headers,
                timeout=self.timeout,
            )
            
            if resp.status_code != 200:
                raise RuntimeError(f"AvalAI API Error {resp.status_code}: {resp.text}")
                
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