"""Coordinate conversion (Qwen3-VL: relative 0..1000, verified in Step 2) and point-answer parsing."""

import pytest

from foveated_search.search.coords import image_to_frame, model_to_frame, model_to_px, px_to_model
from foveated_search.search.parsing import ParseError, parse_point


def test_model_to_px_relative_1000():
    # Step 2 synthetic case: red circle centre (400, 120) on 640x480 is (625, 250) in model coords
    assert model_to_px(625, 250, 640, 480) == pytest.approx((400, 120))
    assert model_to_px(0, 0, 96, 72) == (0, 0)
    assert model_to_px(1000, 1000, 96, 72) == pytest.approx((96, 72))


def test_same_model_point_any_resolution():
    # relative coords do not depend on image size: the centre is (500, 500) everywhere
    for w, h in [(96, 72), (224, 224), (640, 480), (1000, 300)]:
        assert model_to_px(500, 500, w, h) == pytest.approx((w / 2, h / 2))


def test_px_model_roundtrip_and_clamp():
    for w, h in [(640, 480), (96, 72)]:
        for px, py in [(0, 0), (w / 3, h / 4), (w, h)]:
            mx, my = px_to_model(px, py, w, h)
            bx, by = model_to_px(mx, my, w, h)
            assert abs(bx - px) <= w / 1000 and abs(by - py) <= h / 1000
    assert px_to_model(-5, 700, 640, 480) == (0, 1000)


def test_thumbnail_and_crop_to_frame():
    # thumbnail 96x72 of a 640x480 frame: model centre -> frame centre
    assert model_to_frame(500, 500, 96, 72, (0, 0, 640, 480)) == pytest.approx((320, 240))
    # 224x224 crop at (100, 50): crop centre -> (212, 162)
    assert model_to_frame(500, 500, 224, 224, (100, 50, 324, 274)) == pytest.approx((212, 162))
    # a crop resampled to a different sent size still maps through its region
    assert image_to_frame(56, 56, 112, 112, (100, 50, 324, 274)) == pytest.approx((212, 162))


@pytest.mark.parametrize("text,expected", [
    ('{"point": [600, 240]}', (600.0, 240.0)),
    ('{ "point": [127, 274] }', (127.0, 274.0)),                         # structured output style
    ('```json\n{"point": [600, 240]}\n```', (600.0, 240.0)),             # fenced, as the base model answers
    ('```json\n{"point": [138, 172], "label": "remote"}\n```', (138.0, 172.0)),
    ('{"point": null}', None),
    ('[{"point_2d": [10, 20], "label": "cup"}]', (10.0, 20.0)),          # Qwen native grounding format
    ('Sure! {"point": [1, 2]}', (1.0, 2.0)),
])
def test_parse_point_ok(text, expected):
    assert parse_point(text) == expected


@pytest.mark.parametrize("text", [
    '```json\n{"point": [264, 544], "point": null}\n```',   # seen in Step 2: both template options echoed
    "I think it is at the top left",
    '{"point": [1, 2, 3]}',
    '{"point": "left"}',
    '{"box": [1, 2, 3, 4]}',
])
def test_parse_point_rejects(text):
    with pytest.raises(ParseError):
        parse_point(text)
