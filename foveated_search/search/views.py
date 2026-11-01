"""The images the model sees: thumbnails and sharp crops. Shared by eye search (inference) and the
training-data generator (Step 6), so both produce exactly the same pixels for the same inputs."""

from __future__ import annotations

from PIL import Image

from foveated_search.search.geometry import crop_box, thumbnail_size


def resize(frame: Image.Image, size: tuple[int, int]) -> Image.Image:
    return frame if frame.size == size else frame.resize(size, Image.Resampling.LANCZOS)


def make_thumbnail(frame: Image.Image, long_side: int) -> Image.Image:
    """Whole frame, long side snapped to the 32-px grid, short side closest to the true aspect."""
    return resize(frame, thumbnail_size(frame.width, frame.height, long_side))


def make_crop(frame: Image.Image, cx: float, cy: float, size: int) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """Sharp size x size crop of the original frame centred at (cx, cy), clamped inside the frame."""
    box = crop_box(cx, cy, size, frame.width, frame.height)
    return frame.crop(box), box
