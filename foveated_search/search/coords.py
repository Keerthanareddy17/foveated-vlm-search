"""Coordinate conversion between the model's point format and pixels.

Qwen3-VL reports points/boxes as RELATIVE coordinates in [0, scale] (scale = 1000), measured
on the image it was given, independent of the processor's internal resize. Verified in Step 2
(NOTES.md "Coordinate format"). Conversions here use continuous pixel coords (0..W, 0..H).
"""

from __future__ import annotations


def model_to_px(x: float, y: float, width: int, height: int, scale: float = 1000.0) -> tuple[float, float]:
    """Model point (relative, 0..scale) -> pixel coords in the image that was sent."""
    return x / scale * width, y / scale * height


def px_to_model(px: float, py: float, width: int, height: int, scale: float = 1000.0) -> tuple[int, int]:
    """Pixel coords -> model point (rounded ints, clamped to [0, scale]). For training targets."""
    def c(v: float) -> int:
        return int(min(max(round(v), 0), scale))
    return c(px / width * scale), c(py / height * scale)


def image_to_frame(
    px: float, py: float, sent_w: int, sent_h: int, region: tuple[float, float, float, float]
) -> tuple[float, float]:
    """Pixel point in a sent image (a thumbnail or crop of the frame) -> frame pixel coords.

    region = (x0, y0, x1, y1): the frame area the sent image was made from. For a thumbnail
    region = (0, 0, W, H); for a crop it's the crop box (resampled to sent_w x sent_h).
    """
    x0, y0, x1, y1 = region
    return x0 + px * (x1 - x0) / sent_w, y0 + py * (y1 - y0) / sent_h


def model_to_frame(
    x: float, y: float, sent_w: int, sent_h: int, region: tuple[float, float, float, float], scale: float = 1000.0
) -> tuple[float, float]:
    """Model point on a sent image -> frame pixel coords (composition of the two above)."""
    px, py = model_to_px(x, y, sent_w, sent_h, scale)
    return image_to_frame(px, py, sent_w, sent_h, region)
