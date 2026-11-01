"""COCO masks and on-object points.

A box centre often misses the object (donuts, curved or thin objects). The training target point is:
the mask centroid if it lies inside the mask, otherwise the mask pixel farthest from the mask border
(found by repeated 4-neighbour erosion; ties -> the pixel closest to the centroid, then row-major).
Points are pixel centres (x + 0.5, y + 0.5) in frame coordinates.
"""

from __future__ import annotations

import numpy as np
from pycocotools import mask as mask_utils


def seg_to_mask(segmentation, height: int, width: int) -> np.ndarray:
    """Polygon list or (compressed/uncompressed) RLE -> bool mask (H, W)."""
    if isinstance(segmentation, list):
        rles = mask_utils.frPyObjects(segmentation, height, width)
        rle = mask_utils.merge(rles)
    elif isinstance(segmentation.get("counts"), list):
        rle = mask_utils.frPyObjects(segmentation, height, width)
    else:
        rle = segmentation
    return mask_utils.decode(rle).astype(bool)


def _erode(m: np.ndarray) -> np.ndarray:
    out = m.copy()
    out[1:, :] &= m[:-1, :]
    out[:-1, :] &= m[1:, :]
    out[:, 1:] &= m[:, :-1]
    out[:, :-1] &= m[:, 1:]
    out[0, :] = out[-1, :] = False
    out[:, 0] = out[:, -1] = False
    return out


def on_object_point(mask: np.ndarray) -> tuple[float, float] | None:
    """(x, y) pixel centre on the object, or None for an empty mask."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    cx, cy = xs.mean(), ys.mean()
    ix, iy = int(np.floor(cx)), int(np.floor(cy))
    if 0 <= iy < mask.shape[0] and 0 <= ix < mask.shape[1] and mask[iy, ix]:
        return ix + 0.5, iy + 0.5
    # pole of inaccessibility on the pixel grid: erode until empty, keep the last survivors
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    m = np.zeros((y1 - y0 + 2, x1 - x0 + 2), bool)  # 1-px pad so the frame border counts as outside
    m[1:-1, 1:-1] = mask[y0:y1, x0:x1]
    last = m
    while True:
        e = _erode(last)
        if not e.any():
            break
        last = e
    sy, sx = np.nonzero(last)
    d = (sx - 1 + x0 + 0.5 - cx) ** 2 + (sy - 1 + y0 + 0.5 - cy) ** 2
    order = np.lexsort((sx, sy, d))  # closest to centroid, then row-major
    k = order[0]
    return float(sx[k] - 1 + x0) + 0.5, float(sy[k] - 1 + y0) + 0.5


def point_in_mask(mask: np.ndarray, x: float, y: float) -> bool:
    ix, iy = int(np.floor(x)), int(np.floor(y))
    return 0 <= iy < mask.shape[0] and 0 <= ix < mask.shape[1] and bool(mask[iy, ix])
