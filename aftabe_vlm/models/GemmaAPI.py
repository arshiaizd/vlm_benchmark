from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Optional, List
import base64
import os
import requests
import mimetypes
from dataclasses import dataclass

# Adjust imports to match your project
from .base import VisionLanguageModel, ModelResponse

@dataclass
class AvalAIConfig:
    # Use your AvalAI Key (starts with 'aa-...')
    api_key: str = "aa-roEXspUP5ipchJI5JdcDeew7WquYsRMSMJjkwCzRR1QCmoxz" 
    
    # Try "gemma-2-27b-it" or "gemma-3-27b-it"
    model_name: str = "gemma-3-27b-it"
    
    base_url: str = "https://api.avalai.ir/v1"
    timeout: int = 120

class GemmaAPI(VisionLanguageModel):
    """
    Wrapper for using Google's Gemma models via the AvalAI API.
    AvalAI treats Gemma as an OpenAI-compatible chat model.
    """

    def __init__(self, config: Optional[AvalAIConfig] = None) -> None:
        self.config = config or AvalAIConfig()
        
        # 1. Auth & Config
        api_key = self.config.api_key or os.environ.get("AVALAI_API_KEY")
        if not api_key:
            raise RuntimeError("AvalAI API Key is required.")

        self.api_key = api_key
        self.model_name = self.config.model_name
        self.base_url = self.config.base_url.rstrip("/")
        self.timeout = self.config.timeout

        # 2. Endpoint (Standard OpenAI format)
        self.endpoint = f"{self.base_url}/chat/completions"

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    @property
    def name(self) -> str:
        return f"avalai({self.model_name})"

    def _encode_image_as_data_url(self, image_path: str) -> str:
        """
        Encodes image for OpenAI-compatible API (Data URL format).
        Required for AvalAI even when using Google models.
        """
        path = Path(image_path)
        if not path.is_file():
             raise FileNotFoundError(f"Image not found: {path}")
             
        # Detect Mime Type (png, jpeg, webp)
        mime, _ = mimetypes.guess_type(str(path))
        if not mime or not mime.startswith("image/"):
            mime = "image/jpeg" # Fallback

        img_bytes = path.read_bytes()
        b64 = base64.b64encode(img_bytes).decode("utf-8")
        
        # Format: data:image/jpeg;base64,......
        return f"data:{mime};base64,{b64}"

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        image_path: str,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> ModelResponse:
        """Single-turn wrapper (Legacy compatibility)"""
        # Construct a single-turn chat history
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
        Multi-turn chat generation compatible with the main.py loop.
        
        Args:
            messages: List of dicts, e.g.:
                [
                    {"role": "user", "text": "...", "image_path": "..."},
                    {"role": "model", "text": "..."},
                    ...
                ]
        """
        openai_messages = []
        
        # 1. Convert internal message format to OpenAI format
        for msg in messages:
            role = msg["role"]
            content_parts = []
            
            # Map 'model' role to 'assistant' for OpenAI
            if role == "model":
                role = "assistant"
            
            # Add text
            if msg.get("text"):
                content_parts.append({"type": "text", "text": msg["text"]})
            
            # Add image (OpenAI Vision format)
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

        # 2. Build Payload
        payload = {
            "model": self.model_name,
            "messages": openai_messages,
            # "max_tokens": self.max_tokens
        }
        
        # 3. Send Request
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
            
            # Parse response content (handle string vs list)
            if isinstance(content, str):
                raw_text = content
            else:
                # If the API returns a list of content blocks
                raw_text = "\n".join([
                    part.get("text", "") 
                    for part in content 
                    if part.get("type") == "text"
                ])

        except Exception as e:
             raise RuntimeError(f"Metis Request Failed: {e}")

        # 4. Return Standard Response
        provider_payload = {
            "endpoint": self.endpoint,
            "model": self.model_name,
            "raw_response": data,
            "extra_metadata": extra_metadata or {},
        }

        return ModelResponse(raw_text=raw_text, provider_payload=provider_payload)
    