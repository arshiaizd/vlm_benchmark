# file: aftabe_vlm/models/avalai_vlm.py

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
import base64
import os

from openai import OpenAI  # OpenAI-compatible client, but we point it to Aval

from base import VisionLanguageModel, ModelResponse


class Gemma3(VisionLanguageModel):
    """
    VisionLanguageModel implementation that talks to the Aval AI OpenAI-compatible API,
    using the official `openai` Python client.

    IMPORTANT:
    - Requests are sent to Aval, NOT OpenAI:
        base_url = "https://api.avalapis.ir/v1"
      so the endpoint is:
        https://api.avalapis.ir/v1/chat/completions
    - API key: taken from AVALAI_API_KEY env var by default.
    - Vision: image is sent as a base64 data URL via `image_url`.
    """

    def __init__(
        self,
        model_name: str = "gemma-3-27b-it",
        api_key: Optional[str] = None,
        base_url: str = "https://api.avalapis.ir/v1",
        temperature: float = 0.2,
        max_tokens: int = 150,
    ) -> None:
        """
        Args:
            model_name: Aval AI model name (e.g. "cf.gemma-3-12b-it").
            api_key: Aval AI token. If None, read from env var AVALAI_API_KEY.
            base_url: Aval AI base URL (must include /v1 for the OpenAI-style API).
            temperature: sampling temperature.
            max_tokens: maximum tokens for completion.
        """
        if api_key is None:
            api_key = os.environ.get("AVALAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "AvalAIVLM requires an API key. "
                "Set the AVALAI_API_KEY environment variable or pass api_key=..."
            )

        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature

        # OpenAI-compatible client, but pointed at Aval's base_url
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )

        self.name = f"avalai-{model_name}"

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _encode_image_as_data_url(self, image_path: str) -> str:
        """
        Read an image file and return a base64 data URL for use in `image_url`.
        """
        img_bytes = Path(image_path).read_bytes()
        b64 = base64.b64encode(img_bytes).decode("utf-8")
        # PNG is a safe default; Aval’s OpenAI wrapper accepts this.
        return f"data:image/png;base64,{b64}"

    # ------------------------------------------------------------------ #
    # VisionLanguageModel API
    # ------------------------------------------------------------------ #

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        image_path: str,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> ModelResponse:
        """
        Call the Aval AI model once with:
          - system prompt
          - user prompt
          - attached image (base64 data URL)

        Uses the OpenAI-style /chat/completions endpoint on Aval's base_url.
        """
        image_data_url = self._encode_image_as_data_url(image_path)

        user_prompt = system_prompt + "\n" + user_prompt
        messages: list[Dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_data_url,
                            "detail": "auto",
                        },
                    },
                ],
            },
        ]


        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,

            )
        except Exception as e:
            raise RuntimeError(f"AvalAI chat.completions request failed: {e}") from e

        choice = response.choices[0]
        content = choice.message.content

        # Handle both string and list-of-parts content
        if isinstance(content, str):
            raw_text = content
        else:
            fragments: list[str] = []
            for part in content:
                # some providers return dicts, some objects; handle both
                if isinstance(part, dict) and part.get("type") == "text":
                    fragments.append(part.get("text", ""))
                elif hasattr(part, "type") and part.type == "text":
                    fragments.append(getattr(part, "text", ""))
            raw_text = "\n".join(fragments).strip()

        provider_payload: Dict[str, Any] = {
            "provider": "avalai",
            "model": self.model_name,
            "id": getattr(response, "id", None),
            "usage": getattr(response, "usage", None),
            "extra_metadata": extra_metadata or {},
        }

        return ModelResponse(raw_text=raw_text, provider_payload=provider_payload)


if __name__ == "__main__":
    import json

    def main() -> None:
        # --------------------------------------------------
        # 🔧 EDIT THESE VALUES MANUALLY BEFORE RUNNING
        # --------------------------------------------------

        # Path to the image you want to test
        image_path = "../../dataset/en/en_images/1.jpg"

        # Which Aval AI model to use (Gemma on Aval)
        model_name = "gpt-4o-mini"

        # If you want to rely on env AVALAI_API_KEY, set api_key=None.
        api_key = os.environ.get("AVALAI_API_KEY") or "aa-FlbWzVY6khetwVhVvuCnWPW3yl0Hd2vWishfxs2dRFSkTAtc"

        temperature = 0.2
        max_tokens = 500

        system_prompt = (
            "You are an expert multimodal puzzle solver. "
            "Given an image and a user instruction, you must infer a single word "
            "or short phrase that the image represents, and respond in JSON with "
            'keys \"reasoning\" and \"final_answer\".'
        )

        user_prompt = (
            "You are presented with a single image that encodes a word or short phrase.\n"
            "Carefully analyze the image and infer the single intended solution.\n"
            "Return ONLY a JSON object with keys \"reasoning\" and \"final_answer\"."
        )

        # --------------------------------------------------
        # Instantiate the model and call it once
        # --------------------------------------------------
        vlm = Gemma3(
            model_name=model_name,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        print(f"Using model: {vlm.name}")
        print(f"Image path: {image_path}")

        response = vlm.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            image_path=image_path,
            extra_metadata={"source": "avalai_vlm_manual_test"},
        )

        print("\n=== Raw model text ===")
        print(response.raw_text)

        # Try to parse JSON if the model followed the instruction
        try:
            parsed = json.loads(response.raw_text)
            print("\n=== Parsed JSON ===")
            print(json.dumps(parsed, indent=2, ensure_ascii=False))
        except json.JSONDecodeError:
            print("\n(Note: response was not valid JSON; showing raw text only.)")

        print("\n=== Provider payload (meta) ===")
        print(json.dumps(response.provider_payload, indent=2, ensure_ascii=False))

    main()
