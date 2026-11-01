"""Common interface for search strategies: call records, traces, registry, shared call helper.

Every strategy is `async def run(frame, query, client, cfg) -> Trace`, registered by name.
Accounting definitions (CLAUDE.md):
- pixels sampled = source pixels of every image sent, summed over calls, / frame pixels.
  An image sent twice (eye search's thumbnail) counts twice; `pixels_unique_frac` counts each
  distinct image once (what an edge device that caches the thumbnail would send).
- vision tokens = from the real processor (client), summed over calls.
- time_to_answer = sum of per-call total times (sequential search); wall_s = measured wall clock.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Awaitable, Callable

from PIL import Image

from foveated_search.config import Config, load_config
from foveated_search.search.schemas import structured_body
from foveated_search.serving.images import EncodedImage

SEARCH_OVERLAYS = ["configs/search.yaml", "configs/prompts.yaml"]


def search_config(overlays: list[str] | None = None) -> Config:
    cfg = load_config(overlays=[*SEARCH_OVERLAYS, *(overlays or [])])
    for k in ["search.thumbnail_long_side", "search.fixation_size", "search.max_fixations",
              "search.confirm.refine_point", "search.confirm.label",
              "search.sliding_window.overlap", "search.downscale.pixel_budget", "prompts.version", "prompts.point"]:
        cfg.get_key(k)
    return cfg


def prompt(cfg: Config, key: str, **kw: Any) -> str:
    return " ".join(str(cfg["prompts"][key]).split()).format(**kw)


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


@dataclass
class CallRecord:
    role: str                        # point | confirm | tile
    fixation: int | None             # 1-based fixation index (eye search), tile index (sliding window)
    region: list[int]                # frame box [x0, y0, x1, y1] the image was made from
    sent_w: int
    sent_h: int
    resized_w: int | None            # after the processor's resize
    resized_h: int | None
    pixels: int                      # source pixels in the image sent
    png_bytes: int
    image_sha256: str
    vision_tokens: int | None
    prompt_tokens: int | None
    completion_tokens: int | None
    ttft_s: float | None
    total_s: float
    prompt: str
    raw_answer: str
    parsed: Any = None
    parse_error: str | None = None
    finish_reason: str | None = None


@dataclass
class Trace:
    strategy: str
    query: str
    frame_w: int
    frame_h: int
    params: dict
    prompts_version: int
    calls: list[CallRecord] = field(default_factory=list)
    found: bool = False
    final_point: list[float] | None = None     # frame pixels
    fixations: list[list[float]] = field(default_factory=list)  # eye search: frame points looked at
    crops: list[list[int]] = field(default_factory=list)        # eye search crops / sliding-window tiles sent
    repeats: int = 0
    thumbnail_sha256: str | None = None
    evidence: str | None = None
    wall_s: float = 0.0
    meta: dict = field(default_factory=dict)   # harness fields (query_id, image_id, ...)

    @property
    def frame_pixels(self) -> int:
        return self.frame_w * self.frame_h

    def totals(self) -> dict:
        c = self.calls
        uniq = {r.image_sha256: r.pixels for r in c}
        return {
            "n_calls": len(c),
            "pixels_sent": sum(r.pixels for r in c),
            "pixels_sampled_frac": sum(r.pixels for r in c) / self.frame_pixels,
            "pixels_unique_frac": sum(uniq.values()) / self.frame_pixels,
            "png_bytes": sum(r.png_bytes for r in c),
            "vision_tokens": sum(r.vision_tokens or 0 for r in c),
            "prompt_tokens": sum(r.prompt_tokens or 0 for r in c),
            "completion_tokens": sum(r.completion_tokens or 0 for r in c),
            "time_to_answer_s": sum(r.total_s for r in c),
            "parse_errors": sum(r.parse_error is not None for r in c),
        }

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(self.totals())
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"))


async def call(client, cfg: Config, *, role: str, fixation: int | None, image: EncodedImage,
               region: tuple[int, int, int, int], text: str, schema: dict, max_tokens: int) -> CallRecord:
    """One model call: image first, then text; structured output with `schema`."""
    r = await client.chat([image], text, max_tokens=max_tokens,
                          extra_body=structured_body(schema))
    im = r.images[0]
    return CallRecord(
        role=role, fixation=fixation, region=[int(v) for v in region], sent_w=im.width, sent_h=im.height,
        resized_w=im.resized_w, resized_h=im.resized_h, pixels=im.pixels, png_bytes=im.png_bytes,
        image_sha256=sha256(image.png), vision_tokens=im.vision_tokens, prompt_tokens=r.prompt_tokens,
        completion_tokens=r.completion_tokens, ttft_s=r.ttft_s, total_s=r.total_s, prompt=text,
        raw_answer=r.answer, finish_reason=r.finish_reason,
    )


StrategyFn = Callable[[Image.Image, str, Any, Config], Awaitable[Trace]]
REGISTRY: dict[str, StrategyFn] = {}


def register(name: str):
    def deco(fn: StrategyFn) -> StrategyFn:
        if name in REGISTRY:
            raise ValueError(f"strategy {name!r} registered twice")
        REGISTRY[name] = fn
        return fn
    return deco


def get_strategy(name: str) -> StrategyFn:
    import foveated_search.search.strategies  # noqa: F401  (populates REGISTRY)

    if name not in REGISTRY:
        raise KeyError(f"unknown strategy {name!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[name]


async def run_strategy(name: str, frame: Image.Image, query: str, client, cfg: Config) -> Trace:
    t0 = time.perf_counter()
    tr = await get_strategy(name)(frame.convert("RGB"), query, client, cfg)
    tr.wall_s = time.perf_counter() - t0
    return tr


def new_trace(name: str, frame: Image.Image, query: str, cfg: Config, params: dict) -> Trace:
    return Trace(strategy=name, query=query, frame_w=frame.width, frame_h=frame.height, params=params,
                 prompts_version=int(cfg.get_key("prompts.version")))
