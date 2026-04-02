"""All inference backends — BaseBackend, GeminiBackend, OpenAIBackend, TransformersBackend."""

from __future__ import annotations

from abc import ABC, abstractmethod

from PIL import Image
from .request import ImageBlock, InferenceRequest, TextBlock


# ── Base ──────────────────────────────────────────────────────────────────────

class BaseBackend(ABC):
    """Common interface for all inference backends."""

    name: str

    @abstractmethod
    def run(self, request: InferenceRequest) -> str:
        """Run inference and return the model's text response."""
        ...


# ── Gemini ────────────────────────────────────────────────────────────────────

class GeminiBackend(BaseBackend):
    """Native Gemini inference via google-genai SDK."""

    def __init__(
        self,
        name: str,
        model_id: str,
        api_key: str,
        thinking_budget: int | None = None,
    ):
        if not api_key:
            raise ValueError(f"[{name}] GEMINI_API_KEY is not set. Add it to your .env file.")
        self.name = name
        self.model_id = model_id
        self._api_key = api_key
        self._thinking_budget = thinking_budget
        self._client = None

    def _ensure_client(self) -> None:
        """Lazy-load the Gemini client on first use."""
        if self._client is None:
            from google import genai
            self._client = genai.Client(api_key=self._api_key)

    def run(self, request: InferenceRequest) -> str:
        """Generate text output from a multimodal request."""
        from google.genai import types as genai_types
        self._ensure_client()
        parts: list[genai_types.Part] = []

        for block in request.content:
            if isinstance(block, TextBlock):
                parts.append(genai_types.Part.from_text(text=block.text))
            elif isinstance(block, ImageBlock):
                parts.append(
                    genai_types.Part.from_bytes(
                        data=block.read_bytes(),
                        mime_type=block.mime_type(),
                    )
                )

        config_kwargs = {
            "max_output_tokens": request.max_new_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "system_instruction": request.system_prompt or None,
        }
        if self._thinking_budget is not None:
            config_kwargs["thinking_config"] = genai_types.ThinkingConfig(
                thinking_budget=self._thinking_budget
            )
        config = genai_types.GenerateContentConfig(**config_kwargs)
        response = self._client.models.generate_content(
            model=self.model_id, contents=parts, config=config,
        )
        return response.text


# ── OpenAI-compatible ─────────────────────────────────────────────────────────

class OpenAIBackend(BaseBackend):
    """Any service that speaks the OpenAI chat completions API.

    Works with: OpenAI, Anthropic, Gemini (compat), Ollama, MLX-VLM, vLLM.
    Lazy-loads the OpenAI client on first run().
    """

    def __init__(self, name: str, model_id: str, base_url: str, api_key: str | None):
        if not base_url:
            raise ValueError(f"[{name}] base_url is required for OpenAI-compatible backend.")
        self.name = name
        self.model_id = model_id
        self._base_url = base_url
        self._api_key = api_key
        self._client = None

    def _ensure_client(self) -> None:
        """Lazy-load the OpenAI client on first use."""
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(base_url=self._base_url, api_key=self._api_key)

    def run(self, request: InferenceRequest) -> str:
        """Generate text output via OpenAI-compatible chat completions."""
        self._ensure_client()
        content: list[dict] = []
        for block in request.content:
            if isinstance(block, TextBlock):
                content.append({"type": "text", "text": block.text})
            elif isinstance(block, ImageBlock):
                content.append({"type": "image_url", "image_url": {"url": block.as_data_uri()}})

        messages: list[dict] = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": content})

        creation_kwargs = {
            "model": self.model_id,
            "messages": messages,
            "max_tokens": request.max_new_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
        }
        response = self._client.chat.completions.create(**creation_kwargs)
        return response.choices[0].message.content


# ── Transformers (in-process) ─────────────────────────────────────────────────

class TransformersBackend(BaseBackend):
    """Local in-process inference via HuggingFace Transformers."""

    def __init__(
        self,
        name: str,
        hf_model_id: str,
        hf_token: str | None = None,
        hf_cache: str | None = None,
        quantization_level: str | None = None,
    ):
        if not hf_model_id:
            raise ValueError(f"[{name}] hf_model_id is required for Transformers backend.")
        self.name = name
        self.hf_model_id = hf_model_id
        self.hf_token = hf_token
        self.hf_cache = hf_cache
        self.quantization_level = quantization_level.lower() if quantization_level else None
        self.device = self._pick_device()
        self._model = None
        self._processor = None

        self.torch_dtype = self._pick_torch_dtype()

        if self.quantization_level not in {None, "4bit"}:
            raise ValueError(
                f"[{name}] quantization_level must be one of: None, 4bit. "
                f"Got: {self.quantization_level!r}"
            )

    @staticmethod
    def _pick_device() -> str:
        """Detect best available device: CUDA > MPS > CPU."""
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _pick_torch_dtype(self):
        """Pick compute dtype based on runtime device capabilities."""
        import torch

        if self.device == "mps":
            return torch.bfloat16

        if self.device == "cpu":
            return torch.float32

        # CUDA: prefer bfloat16, fall back to float16 on older GPUs.
        current_device = torch.cuda.current_device()
        major, _minor = torch.cuda.get_device_capability(current_device)
        if major < 8:
            print(f"[Info] GPU {current_device} lacks native bfloat16 support; falling back to float16.")
            return torch.float16
        return torch.bfloat16

    def _build_quantization_config(self):
        """Build bitsandbytes quantization config for 4-bit mode."""
        if self.quantization_level is None:
            return None

        if self.device != "cuda":
            print(
                f"[{self.name}] quantization_level={self.quantization_level} requested, "
                f"but device '{self.device}' does not support bitsandbytes quantization. "
                "Falling back to non-quantized loading."
            )
            return None

        from transformers import BitsAndBytesConfig
        if self.quantization_level == "4bit":
            print(f"Configuration: 4-bit (NF4) | Compute dtype: {self.torch_dtype}")
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self.torch_dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        return None

    def _ensure_loaded(self) -> None:
        """Lazy-load processor and model on first use."""
        if self._model is not None:
            return

        from transformers import AutoModelForImageTextToText, AutoProcessor

        print(f"[{self.name}] Loading processor for '{self.hf_model_id}' …")
        self._processor = AutoProcessor.from_pretrained(
            self.hf_model_id, token=self.hf_token, cache_dir=self.hf_cache,
        )
        print(
            f"[{self.name}] Loading model on {self.device} "
            f"(dtype={self.torch_dtype}, quantization={self.quantization_level}) …"
        )

        quantization_config = self._build_quantization_config()
        load_kwargs = {
            "token": self.hf_token,
            "cache_dir": self.hf_cache,
            "dtype": self.torch_dtype,
            "trust_remote_code": True,
        }
        if quantization_config is not None:
            load_kwargs["device_map"] = "auto"
            load_kwargs["quantization_config"] = quantization_config

        self._model = AutoModelForImageTextToText.from_pretrained(
            self.hf_model_id,
            **load_kwargs,
        )
        if quantization_config is None:
            self._model = self._model.to(self.device)

        self._model.eval()
        actual_dtype = next(self._model.parameters()).dtype
        print(f"[{self.name}] Model loaded on {self.device} | dtype: {actual_dtype}\n")

    def _image_content_entry(self, image: Image.Image) -> dict:
        if "qwen" in self.hf_model_id.lower() and "vl" in self.hf_model_id.lower():
            return {"type": "image", "image": image}
        return {"type": "image"}

    def run(self, request: InferenceRequest) -> str:
        """Generate text output using local Transformers model."""
        import torch
        self._ensure_loaded()
        all_pil_images: list[Image.Image] = []
        content_list: list[dict] = []

        for block in request.content:
            if isinstance(block, TextBlock):
                content_list.append({"type": "text", "text": block.text})
            elif isinstance(block, ImageBlock):
                pil = block.load()
                all_pil_images.append(pil)
                content_list.append(self._image_content_entry(pil))

        messages: list[dict] = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": content_list})

        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        processor_kwargs: dict = {"text": [text], "return_tensors": "pt"}
        if all_pil_images:
            processor_kwargs["images"] = all_pil_images
        inputs = self._processor(**processor_kwargs).to(self.device)

        generation_kwargs = {
            "max_new_tokens": request.max_new_tokens,
            "do_sample": request.do_sample,
            "temperature": request.temperature if request.do_sample else None,
            "top_p": request.top_p if request.do_sample else None,
        }
        with torch.no_grad():
            output_ids = self._model.generate(**inputs, **generation_kwargs)

        new_ids = output_ids[:, inputs["input_ids"].shape[-1]:]
        return self._processor.batch_decode(new_ids, skip_special_tokens=True)[0]
