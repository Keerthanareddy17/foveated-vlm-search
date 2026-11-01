"""Lossless image encoding for requests. The bytes we send are exactly the pixels the model gets."""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass

from PIL import Image


@dataclass(frozen=True)
class EncodedImage:
    png: bytes
    width: int
    height: int

    @property
    def pixels(self) -> int:
        return self.width * self.height

    @property
    def num_bytes(self) -> int:
        return len(self.png)

    def data_url(self) -> str:
        return "data:image/png;base64," + base64.b64encode(self.png).decode("ascii")

    def decode(self) -> Image.Image:
        """The image exactly as the server will decode it."""
        return Image.open(io.BytesIO(self.png)).convert("RGB")


def encode_png(img: Image.Image, compress_level: int = 6) -> EncodedImage:
    if img.mode != "RGB":
        img = img.convert("RGB")
    buf = io.BytesIO()
    # optimize=False + fixed level -> deterministic bytes for the same pixels.
    img.save(buf, format="PNG", compress_level=compress_level, optimize=False)
    return EncodedImage(png=buf.getvalue(), width=img.width, height=img.height)
