"""Load the frame for an eval row: a COCO eval image, or a Step 8b mosaic (rendered on first use)."""

from __future__ import annotations

from PIL import Image

from foveated_search.config import Config
from foveated_search.data.fetch_images import eval_image_path


def load_frame(cfg: Config, row: dict) -> Image.Image:
    if "grid" in row:
        from foveated_search.data.mosaic import load_mosaic

        return load_mosaic(cfg, row)
    return Image.open(eval_image_path(cfg, row["file_name"])).convert("RGB")
