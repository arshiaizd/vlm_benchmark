from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Optional, List
import base64
import os
import requests

# Adjust these imports to match your project structure
# from validation.prompts_config import get_base_prompts, get_prompt_variants
from .base import VisionLanguageModel, ModelResponse


class OpenaiGPT(VisionLanguageModel):
    """
    VisionLanguageModel implementation that talks to the Metis wrapper API.
    The API is treated as OpenAI-compatible chat/completions with vision support.
    """

    def __init__(
        self,
        api_key: Optional[str] = "aa-roEXspUP5ipchJI5JdcDeew7WquYsRMSMJjkwCzRR1QCmoxz", # Make sure to set this or env var
        model: str = "gpt-5.2",
        base_url: str = "https://api.avalapis.ir/v1", # Common Metis base URL, check yours
        provider: str = "openai_chat_completion",
        # max_tokens: int = 1000,
        effort: str = 'none',
        timeout: int = 120,
    ):
        if api_key is None:
            api_key = os.environ.get("METIS_API_KEY") or "YOUR_FALLBACK_KEY" # Replace if needed
            
        if not api_key:
            raise RuntimeError("requires an API key.")

        self.api_key = api_key
        self.effort = effort
        self.model_name = model
        self.base_url = base_url.rstrip("/")
        self.provider = provider
        # self.max_tokens = max_tokens
        self.timeout = timeout

        # Metis usually mirrors OpenAI paths directly
        # If your provider URL is different, adjust this endpoint construction
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
        # Guess mime type or default to png
        return f"data:image/jpeg;base64,{b64}"

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
        
        # Add 'reasoning_effort' only for models that support it (like o1/gpt-4o-reasoning)
        if self.effort:
            payload["reasoning_effort"] = self.effort

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