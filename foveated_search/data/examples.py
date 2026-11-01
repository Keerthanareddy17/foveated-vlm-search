"""Single example images for smoke tests / benchmarks (Step 2). Not the eval dataset (Step 3)."""

from __future__ import annotations

import urllib.request
from pathlib import Path

from PIL import Image

from foveated_search.config import Config

# Two cats on a pink couch with two remotes, 640x480.
COCO_EXAMPLE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"


def fetch_coco_example(cfg: Config, url: str = COCO_EXAMPLE_URL) -> Image.Image:
    """Download once into <data_dir>/examples/ on Drive, then reuse."""
    cache = Path(cfg.get_key("paths.data_dir")) / "examples" / Path(url).name
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=60) as r:
            cache.write_bytes(r.read())
    return Image.open(cache).convert("RGB")
