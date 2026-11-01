"""Image geometry for search: thumbnail / downscale sizes on the 32-px grid, crops, tiles.

Boxes are (x0, y0, x1, y1) in frame pixels, x1/y1 exclusive. Qwen3-VL's processor snaps image
sides to multiples of 32 (patch 16 x merge 2, Step 2), so we pick such sizes ourselves and
the model sees exactly what we sent.
"""

from __future__ import annotations

import math

GRID = 32  # patch_size * merge_size for Qwen3-VL


def _snap(v: float, grid: int = GRID) -> int:
    return max(grid, int(round(v / grid)) * grid)


def thumbnail_size(width: int, height: int, long_side: int, grid: int = GRID) -> tuple[int, int]:
    """Long side snapped to the grid; short side = the grid multiple closest to the true aspect."""
    L = _snap(long_side, grid)
    if width >= height:
        return L, _snap(L * height / width, grid)
    return _snap(L * width / height, grid), L


def downscale_size(width: int, height: int, pixel_budget: int, grid: int = GRID) -> tuple[int, int]:
    """Largest grid-aligned (w, h) with w*h <= budget, choosing the closest aspect among the largest.
    Never upscales beyond the frame's own grid-snapped size."""
    aspect = width / height
    best = None
    max_w = max(grid, (width // grid) * grid)
    max_h = max(grid, (height // grid) * grid)
    for w in range(grid, max_w + 1, grid):
        for h in (math.floor(w / aspect / grid) * grid, math.ceil(w / aspect / grid) * grid):
            if h < grid or h > max_h or w * h > pixel_budget:
                continue
            key = (w * h, -abs(w / h - aspect))
            if best is None or key > best[0]:
                best = (key, (w, h))
    if best is None:
        return grid, grid
    return best[1]


def crop_box(cx: float, cy: float, size: int, width: int, height: int) -> tuple[int, int, int, int]:
    """size x size box centred at (cx, cy), shifted to lie inside the frame. If the frame is smaller
    than `size` in a dimension, the crop spans that whole dimension."""
    cw, ch = min(size, width), min(size, height)
    x0 = int(round(cx - cw / 2))
    y0 = int(round(cy - ch / 2))
    x0 = min(max(x0, 0), width - cw)
    y0 = min(max(y0, 0), height - ch)
    return x0, y0, x0 + cw, y0 + ch


def _starts(length: int, tile: int, stride: int) -> list[int]:
    if length <= tile:
        return [0]
    s = list(range(0, length - tile + 1, stride))
    if s[-1] != length - tile:
        s.append(length - tile)  # last tile shifted inward, never padded
    return s


def tiles(width: int, height: int, tile: int, overlap: float, order: str = "raster") -> list[tuple[int, int, int, int]]:
    """Tiles covering every pixel. Stride = tile * (1 - overlap)."""
    if not 0 <= overlap < 1:
        raise ValueError("overlap must be in [0, 1)")
    stride = max(1, int(round(tile * (1 - overlap))))
    tw, th = min(tile, width), min(tile, height)
    boxes = [(x, y, x + tw, y + th) for y in _starts(height, tile, stride) for x in _starts(width, tile, stride)]
    if order == "raster":
        return boxes
    if order == "center_out":
        cx, cy = width / 2, height / 2
        return sorted(boxes, key=lambda b: ((b[0] + b[2]) / 2 - cx) ** 2 + ((b[1] + b[3]) / 2 - cy) ** 2)
    raise ValueError(f"unknown tile order {order!r}")


def in_box(x: float, y: float, box: tuple[float, float, float, float]) -> bool:
    return box[0] <= x <= box[2] and box[1] <= y <= box[3]


def in_any_xywh(x: float, y: float, boxes: list[list[float]]) -> bool:
    """Ground-truth check: point inside any COCO [x, y, w, h] box (edges inclusive)."""
    return any(bx <= x <= bx + bw and by <= y <= by + bh for bx, by, bw, bh in boxes)
