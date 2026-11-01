"""GPU-free tests for image encoding, request building, usage parsing and the token helper."""

import base64
import io
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from foveated_search.serving.client import CallResult, SentImage, build_messages, parse_usage
from foveated_search.serving.images import encode_png
from foveated_search.serving.vision_tokens import tokens_from_grid


def _noise(w, h, seed=0):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8), "RGB")


def test_png_is_lossless_and_sized():
    img = _noise(97, 61)
    e = encode_png(img)
    assert (e.width, e.height, e.pixels) == (97, 61, 97 * 61)
    assert e.num_bytes == len(e.png) > 0
    assert np.array_equal(np.asarray(e.decode()), np.asarray(img))


def test_png_deterministic_and_converts_mode():
    img = _noise(32, 32)
    assert encode_png(img).png == encode_png(img.copy()).png
    rgba = img.convert("RGBA")
    assert np.array_equal(np.asarray(encode_png(rgba).decode()), np.asarray(img))


def test_data_url_roundtrip():
    e = encode_png(_noise(8, 8))
    url = e.data_url()
    assert url.startswith("data:image/png;base64,")
    raw = base64.b64decode(url.split(",", 1)[1])
    assert raw == e.png and Image.open(io.BytesIO(raw)).size == (8, 8)


def test_build_messages_images_first_then_text():
    a, b = encode_png(_noise(8, 8)), encode_png(_noise(16, 8, 1))
    msgs = build_messages([a, b], "point to the cup", system="sys")
    assert msgs[0] == {"role": "system", "content": "sys"}
    content = msgs[1]["content"]
    assert [c["type"] for c in content] == ["image_url", "image_url", "text"]
    assert content[2]["text"] == "point to the cup"
    assert build_messages([], "hi")[0]["content"] == [{"type": "text", "text": "hi"}]


def test_parse_usage_object_dict_none():
    assert parse_usage(SimpleNamespace(prompt_tokens=54, completion_tokens=9)) == (54, 9)
    assert parse_usage({"prompt_tokens": 3, "completion_tokens": 1}) == (3, 1)
    assert parse_usage(None) == (None, None)


def test_call_result_totals():
    r = CallResult("x", 10, 2, 15, 0.1, 0.2, "stop",
                   [SentImage(96, 72, 96 * 72, 1000, 96, 64, 6), SentImage(224, 224, 224 * 224, 5000, 224, 224, 49)])
    d = r.to_dict()
    assert d["pixels_sent"] == 96 * 72 + 224 * 224
    assert d["bytes_sent"] == 6000


@pytest.mark.parametrize(
    "grid,merge,expected",
    [((1, 4, 6), 2, 6),        # 96x64 image, patch 16 -> 6x4 patches -> 6 tokens (measured in Step 2)
     ((1, 14, 14), 2, 49),     # 224x224
     ((1, 30, 40), 2, 300),    # 640x480
     ((1, 14, 20), 2, 70)],    # 320x224 (default min_pixels upscaling of a 96 px thumbnail)
)
def test_tokens_from_grid(grid, merge, expected):
    assert tokens_from_grid(grid, merge) == expected


def test_tokens_from_grid_rejects_unmergeable():
    with pytest.raises(ValueError):
        tokens_from_grid((1, 3, 4), 2)
