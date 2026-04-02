"""Backend factory — instantiates the correct backend from a config client dict."""

from __future__ import annotations

import os
from dotenv import load_dotenv
from .backends import BaseBackend, GeminiBackend, OpenAIBackend, TransformersBackend

load_dotenv()

_hf_token = os.getenv("HF_TOKEN")
_hf_home = os.getenv("HF_HOME")  # None = use HuggingFace default (~/.cache/huggingface)


def get_backend_from_config(client: dict) -> BaseBackend:
    """Instantiate the correct backend from a config client dict."""
    name    = client["name"]
    backend = client["backend"]
    model_id = client["model_id"]

    api_key_env = client.get("api_key_env")
    api_key = os.getenv(api_key_env) if api_key_env else None

    if backend == "gemini":
        return GeminiBackend(
            name=name,
            model_id=model_id,
            api_key=api_key,
            thinking_budget=client.get("thinking_budget"),
        )

    if backend == "openai":
        base_url = client.get("base_url", "https://api.openai.com/v1")
        return OpenAIBackend(name=name, model_id=model_id, base_url=base_url, api_key=api_key)

    if backend == "transformer":
        return TransformersBackend(
            name=name,
            hf_model_id=model_id,
            hf_token=_hf_token,
            hf_cache=_hf_home,
            quantization_level=client.get("quantization_level"),
        )

    raise ValueError(
        f"Unknown backend '{backend}' for client '{name}'. "
        "Expected one of: gemini, openai, transformer."
    )
