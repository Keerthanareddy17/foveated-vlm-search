"""Async client for the vLLM OpenAI-compatible server, with token counts and timings.

Timing: time.perf_counter on the client. The client runs on the same VM as the server
(127.0.0.1), so this is effectively server-side latency (no WAN round trip).
TTFT = time until the first streamed chunk that carries content.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from openai import AsyncOpenAI
from PIL import Image

from foveated_search.config import Config
from foveated_search.serving.images import EncodedImage, encode_png
from foveated_search.serving.launch import base_url
from foveated_search.serving.vision_tokens import VisionTokenCounter, get_counter


@dataclass
class SentImage:
    width: int
    height: int
    pixels: int
    png_bytes: int
    resized_w: int | None = None
    resized_h: int | None = None
    vision_tokens: int | None = None


@dataclass
class CallResult:
    answer: str
    prompt_tokens: int | None
    completion_tokens: int | None
    vision_tokens: int | None  # sum over images, from the real processor
    ttft_s: float | None
    total_s: float
    finish_reason: str | None
    images: list[SentImage] = field(default_factory=list)
    cached_tokens: int | None = None  # prefix-cache hits (server flag --enable-prompt-tokens-details)

    @property
    def pixels_sent(self) -> int:
        return sum(i.pixels for i in self.images)

    @property
    def bytes_sent(self) -> int:
        return sum(i.png_bytes for i in self.images)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["pixels_sent"], d["bytes_sent"] = self.pixels_sent, self.bytes_sent
        return d


def build_messages(images: list[EncodedImage], text: str, system: str | None = None) -> list[dict]:
    """Images first, then text (Qwen convention). Extra text (e.g. already-checked points) goes in `text`."""
    content: list[dict] = [{"type": "image_url", "image_url": {"url": im.data_url()}} for im in images]
    content.append({"type": "text", "text": text})
    msgs: list[dict] = [{"role": "system", "content": system}] if system else []
    msgs.append({"role": "user", "content": content})
    return msgs


def parse_cached(usage: Any) -> int | None:
    if usage is None:
        return None
    det = usage.get("prompt_tokens_details") if isinstance(usage, dict) else getattr(usage, "prompt_tokens_details", None)
    if det is None:
        return None
    return det.get("cached_tokens") if isinstance(det, dict) else getattr(det, "cached_tokens", None)


def parse_usage(usage: Any) -> tuple[int | None, int | None]:
    if usage is None:
        return None, None
    get = usage.get if isinstance(usage, dict) else lambda k: getattr(usage, k, None)
    return get("prompt_tokens"), get("completion_tokens")


class VLMClient:
    def __init__(self, cfg: Config, counter: VisionTokenCounter | None | bool = True):
        self.cfg = cfg
        c = cfg["client"]
        self.model = cfg.get_key("model.served_name")
        self.temperature = float(c["temperature"])
        self.max_tokens = int(c["max_tokens"])
        self.png_level = int(c["png_compress_level"])
        self.seed = int(cfg["seed"])
        self.oa = AsyncOpenAI(base_url=base_url(cfg) + "/v1", api_key="EMPTY", timeout=float(c["timeout_s"]))
        if counter is True:
            counter = get_counter(
                str(cfg.get_key("model.id")),
                cfg.get_key("server.image_min_pixels"),
                cfg.get_key("server.image_max_pixels"),
            )
        self.counter: VisionTokenCounter | None = counter or None

    def encode(self, img: Image.Image) -> EncodedImage:
        return encode_png(img, self.png_level)

    async def aencode(self, img: Image.Image) -> EncodedImage:
        """encode() off the event loop: PNG encoding of a full frame costs ~90 ms of CPU (PIL releases the GIL),
        which would otherwise serialize concurrent searches (Step 8)."""
        return await asyncio.to_thread(self.encode, img)

    async def chat(
        self,
        images: list[Image.Image | EncodedImage],
        text: str,
        *,
        max_tokens: int | None = None,
        system: str | None = None,
        response_format: dict | None = None,
        extra_body: dict | None = None,
    ) -> CallResult:
        enc = [im if isinstance(im, EncodedImage) else await self.aencode(im) for im in images]
        sent = [SentImage(e.width, e.height, e.pixels, e.num_bytes) for e in enc]
        if self.counter is not None:
            for s, e in zip(sent, enc):
                info = self.counter.count(e.decode())  # exactly what the server decodes
                s.resized_w, s.resized_h, s.vision_tokens = info.resized_w, info.resized_h, info.tokens
        kwargs: dict[str, Any] = dict(
            model=self.model,
            messages=build_messages(enc, text, system),
            max_tokens=max_tokens or self.max_tokens,
            temperature=self.temperature,
            seed=self.seed,
            stream=True,
            stream_options={"include_usage": True},
        )
        if response_format is not None:
            kwargs["response_format"] = response_format
        if extra_body:
            kwargs["extra_body"] = extra_body

        t0 = time.perf_counter()
        ttft = None
        parts: list[str] = []
        finish = None
        usage = None
        stream = await self.oa.chat.completions.create(**kwargs)
        async for chunk in stream:
            if chunk.usage is not None:
                usage = chunk.usage
            for ch in chunk.choices:
                if ch.delta and ch.delta.content:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    parts.append(ch.delta.content)
                if ch.finish_reason:
                    finish = ch.finish_reason
        total = time.perf_counter() - t0
        pt, ct = parse_usage(usage)
        vt = sum(s.vision_tokens for s in sent) if sent and all(s.vision_tokens is not None for s in sent) else (0 if not sent else None)
        return CallResult("".join(parts), pt, ct, vt, ttft, total, finish, sent, parse_cached(usage))

    async def close(self) -> None:
        await self.oa.close()
