"""Data structures shared by all backends — what you send to a model."""

from __future__ import annotations

import base64
import mimetypes
from dataclasses import dataclass
from PIL import Image


@dataclass
class TextBlock:
    """A plain text segment in the prompt."""
    text: str


@dataclass
class ImageBlock:
    """A local image file in the prompt."""
    path: str

    def load(self) -> Image.Image:
        with Image.open(self.path) as img:
            return img.convert("RGB")

    def read_bytes(self) -> bytes:
        with open(self.path, "rb") as f:
            return f.read()

    def mime_type(self) -> str:
        mime, _ = mimetypes.guess_type(self.path)
        return mime or "image/jpeg"

    def as_data_uri(self) -> str:
        b64 = base64.b64encode(self.read_bytes()).decode("utf-8")
        return f"data:{self.mime_type()};base64,{b64}"


ContentBlock = TextBlock | ImageBlock


@dataclass
class InferenceRequest:
    """Ordered list of text/image blocks — same structure for every backend.

    Usage:
        # Single image + instruction
        req = InferenceRequest(content=[
            ImageBlock("input/images/image_1.png"),
            TextBlock("What is the core concept shown here?"),
        ], system_prompt="", max_new_tokens=4096, temperature=1.0, top_p=1.0)

        # Multiple images + instruction
        req = InferenceRequest(content=[
            ImageBlock("input/images/image_1.png"),
            ImageBlock("input/images/image_2.png"),
            TextBlock("Compare the two slides."),
        ], system_prompt="", max_new_tokens=4096, temperature=1.0, top_p=1.0)
    """

    content: list[ContentBlock]
    system_prompt: str = ""
    max_new_tokens: int = 4096
    temperature: float = 1.0
    top_p: float = 1.0

    @property
    def do_sample(self) -> bool:
        return self.temperature > 0.0

    @property
    def has_images(self) -> bool:
        return any(isinstance(b, ImageBlock) for b in self.content)

    def image_blocks(self) -> list[ImageBlock]:
        return [b for b in self.content if isinstance(b, ImageBlock)]
