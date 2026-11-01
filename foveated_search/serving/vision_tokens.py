"""Vision-token counts from the model's real processor (not computed by hand).

The processor resizes each image (smart_resize: sides rounded to multiples of patch*merge,
total pixels clamped to [min_pixels, max_pixels]), returns image_grid_thw = (t, h, w) in
patches, and the LLM sees t*h*w / merge_size**2 image tokens (plus <vision_start>/<vision_end>).
Use the SAME min/max pixels as the server (configs: server.image_min_pixels / image_max_pixels).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from PIL import Image


def tokens_from_grid(grid_thw: tuple[int, int, int] | list[int], merge_size: int) -> int:
    t, h, w = (int(v) for v in grid_thw)
    if h % merge_size or w % merge_size:
        raise ValueError(f"grid {grid_thw} not divisible by merge_size {merge_size}")
    return t * h * w // (merge_size**2)


@dataclass(frozen=True)
class VisionTokenInfo:
    sent_w: int
    sent_h: int
    resized_w: int
    resized_h: int
    grid_thw: tuple[int, int, int]
    tokens: int


class VisionTokenCounter:
    def __init__(self, model_id: str, min_pixels: int | None = None, max_pixels: int | None = None):
        from transformers import AutoProcessor  # heavy import, only when used

        self.processor = AutoProcessor.from_pretrained(model_id)
        ip = self.processor.image_processor
        self.patch_size = int(ip.patch_size)
        self.merge_size = int(ip.merge_size)
        if (min_pixels is None) != (max_pixels is None):
            raise ValueError("pass both min_pixels and max_pixels, or neither")
        self.size = (
            {"shortest_edge": int(min_pixels), "longest_edge": int(max_pixels)}
            if min_pixels is not None
            else dict(ip.size)  # model defaults from preprocessor_config.json
        )

    def count(self, img: Image.Image) -> VisionTokenInfo:
        out = self.processor.image_processor(images=[img.convert("RGB")], size=self.size, return_tensors="pt")
        thw = tuple(int(v) for v in out["image_grid_thw"][0].tolist())
        return VisionTokenInfo(
            sent_w=img.width,
            sent_h=img.height,
            resized_w=thw[2] * self.patch_size,
            resized_h=thw[1] * self.patch_size,
            grid_thw=thw,
            tokens=tokens_from_grid(thw, self.merge_size),
        )


@lru_cache(maxsize=4)
def get_counter(model_id: str, min_pixels: int | None, max_pixels: int | None) -> VisionTokenCounter:
    return VisionTokenCounter(model_id, min_pixels, max_pixels)
