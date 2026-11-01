"""Step 4 search strategies, GPU-free: geometry, conversions, tiling, loops with a scripted client,
schemas, and accounting."""

import asyncio
import json

import numpy as np
import pytest
from PIL import Image

from foveated_search.search.base import Trace, run_strategy, search_config
from foveated_search.search.coords import model_to_frame, px_to_model
from foveated_search.search.geometry import crop_box, downscale_size, in_any_xywh, thumbnail_size, tiles
from foveated_search.search.parsing import ParseError
from foveated_search.search.schemas import confirm_schema, parse_confirm, parse_p, point_schema, structured_body
from foveated_search.search.strategies import downscale_budget
from tests.mock_client import MockClient

CFG = search_config()


def frame(w=640, h=480, seed=0):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8), "RGB")


def run(name, img, query, client, cfg=CFG) -> Trace:
    return asyncio.run(run_strategy(name, img, query, client, cfg))


def cfg_with(**search_overrides):
    cfg = search_config()
    for dotted, v in search_overrides.items():
        node = cfg["search"]
        *parents, leaf = dotted.split("__")
        for p in parents:
            node = node[p]
        node[leaf] = v
    return cfg


# ------------------------------------------------------------------ geometry
@pytest.mark.parametrize("w,h,L,exp", [
    (640, 480, 96, (96, 64)),     # 72 -> 64 is closer to 4:3 than 96
    (480, 640, 96, (64, 96)),
    (640, 427, 96, (96, 64)),
    (640, 640, 96, (96, 96)),
    (640, 480, 128, (128, 96)),   # exact 4:3
    (640, 169, 96, (96, 32)),     # very wide: short side never below 32
    (640, 480, 100, (96, 64)),    # long side snapped to the grid
])
def test_thumbnail_size_grid_and_aspect(w, h, L, exp):
    tw, th = thumbnail_size(w, h, L)
    assert (tw, th) == exp and tw % 32 == 0 and th % 32 == 0


def test_downscale_size_fits_budget_and_grid():
    for w, h in [(640, 480), (480, 640), (640, 427), (500, 375)]:
        for budget in [6144 + 50176, 20000, 100000]:
            dw, dh = downscale_size(w, h, budget)
            assert dw * dh <= budget and dw % 32 == 0 and dh % 32 == 0
            assert abs(dw / dh - w / h) < 0.35
    assert downscale_size(640, 480, 56320) == (288, 192)  # 55296 px, closest to budget


def test_downscale_default_budget_is_thumb_plus_fixation():
    assert downscale_budget(640, 480, CFG) == 96 * 64 + 224 * 224
    assert downscale_budget(640, 169, CFG) == 96 * 32 + 224 * 169  # crop clamped to frame height


@pytest.mark.parametrize("cx,cy,exp", [
    (320, 240, (208, 128, 432, 352)),   # centred
    (0, 240, (0, 128, 224, 352)),       # left edge
    (640, 240, (416, 128, 640, 352)),   # right edge
    (320, 0, (208, 0, 432, 224)),       # top edge
    (320, 480, (208, 256, 432, 480)),   # bottom edge
    (0, 0, (0, 0, 224, 224)),           # corners
    (640, 0, (416, 0, 640, 224)),
    (0, 480, (0, 256, 224, 480)),
    (640, 480, (416, 256, 640, 480)),
])
def test_crop_box_centering_and_clamping(cx, cy, exp):
    assert crop_box(cx, cy, 224, 640, 480) == exp


def test_crop_box_frame_smaller_than_crop():
    assert crop_box(100, 50, 224, 300, 169) == (0, 0, 224, 169)       # height smaller
    assert crop_box(50, 50, 224, 200, 150) == (0, 0, 200, 150)        # both smaller: whole frame


def test_tiles_cover_every_pixel_and_overlap():
    for w, h in [(640, 480), (500, 333), (224, 224), (200, 640)]:
        for overlap in [0.0, 0.25, 0.5]:
            boxes = tiles(w, h, 224, overlap)
            cover = np.zeros((h, w), bool)
            for x0, y0, x1, y1 in boxes:
                assert 0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h       # never padded outside
                cover[y0:y1, x0:x1] = True
            assert cover.all()
    xs = sorted({b[0] for b in tiles(640, 480, 224, 0.25)})
    assert xs == [0, 168, 336, 416]                                   # stride 168, last shifted inward
    assert len(tiles(640, 480, 224, 0.25)) == 12
    assert tiles(640, 480, 224, 0.25)[0] == (0, 0, 224, 224)          # raster: top-left first
    co = tiles(640, 480, 224, 0.25, "center_out")[0]
    assert co[0] <= 320 <= co[2] and co[1] <= 240 <= co[3]


# ------------------------------------------------------------------ coordinate conversion
def test_thumbnail_to_frame_with_grid_resize():
    # 640x480 frame -> 96x64 thumbnail (aspect not exact): relative coords still map to the frame
    tw, th = thumbnail_size(640, 480, 96)
    assert model_to_frame(500, 500, tw, th, (0, 0, 640, 480)) == pytest.approx((320, 240))
    assert model_to_frame(250, 750, tw, th, (0, 0, 640, 480)) == pytest.approx((160, 360))


def test_crop_and_tile_to_frame():
    box = crop_box(100, 100, 224, 640, 480)                 # clamped to (0, 0, 224, 224)
    assert model_to_frame(500, 500, 224, 224, box) == pytest.approx((112, 112))
    tile = (416, 256, 640, 480)
    x, y = model_to_frame(*px_to_model(100, 50, 224, 224), 224, 224, tile)
    assert (x, y) == pytest.approx((516, 306), abs=0.3)


# ------------------------------------------------------------------ schemas + parsing
def test_parse_point_answers():
    assert parse_p('{"point_2d":[12,34]}') == (12.0, 34.0)
    assert parse_p('{"point_2d": [12, 34]}') == (12.0, 34.0)
    assert parse_p('{ "absent": true }') is None
    for bad in ['{"point_2d":[1,2],"point_2d":[3,4]}', '{"point":[1,2]}', '{"point_2d":[1]}', '{"point_2d":"x"}',
                "nope", '{"point_2d":[true,1]}', '{"absent":false}', '{"point_2d":[1,2],"absent":true}', '{"p":[1,2]}']:
        with pytest.raises(ParseError):
            parse_p(bad)


def test_parse_confirm_answers():
    assert parse_confirm('{"y":false}').yes is False
    a = parse_confirm('{"y":true,"point_2d":[500,400],"l":"red apple"}')
    assert (a.yes, a.point, a.label) == (True, (500.0, 400.0), "red apple")
    assert parse_confirm('{"y":true}').point is None
    for bad in ['{"y":"yes"}', '{"y":true,"y":false}', '{"y":true,"point_2d":[1,2,3]}', "", '{"y":true,"l":5}']:
        with pytest.raises(ParseError):
            parse_confirm(bad)


def test_schemas_are_minimal_and_switchable():
    pt = point_schema()["anyOf"]
    assert pt[0]["required"] == ["point_2d"] and pt[1]["required"] == ["absent"]
    full = confirm_schema(True, True, 24)["anyOf"]
    assert full[0]["required"] == ["y"] and full[1]["required"] == ["y", "point_2d", "l"]
    assert full[1]["properties"]["l"]["maxLength"] == 24
    assert confirm_schema(False, False, 24)["anyOf"][1]["required"] == ["y"]
    assert confirm_schema(True, False, 24)["anyOf"][1]["required"] == ["y", "point_2d"]
    assert structured_body(point_schema()) == {"structured_outputs": {"json": point_schema()}}


# ------------------------------------------------------------------ strategies with a scripted client
def test_full_frame_and_downscale():
    img = frame()
    t = run("full_frame", img, "cup", MockClient(['{"point_2d":[500,250]}']))
    assert t.found and t.final_point == [320.0, 120.0] and t.totals()["pixels_sampled_frac"] == 1.0
    m = MockClient(['{"absent":true}'])
    t = run("downscale", img, "cup", m)
    assert not t.found and t.final_point is None
    assert m.requests[0]["images"][0].width == 288 and m.requests[0]["images"][0].height == 192
    assert t.totals()["pixels_sampled_frac"] == pytest.approx(288 * 192 / (640 * 480))


PT = cfg_with(sliding_window__tile_mode="point")


def test_sliding_window_stops_at_first_hit():
    img = frame()
    m = MockClient(['{"absent":true}'] * 5 + ['{"point_2d":[500,500]}'])
    t = run("sliding_window", img, "cup", m, PT)
    assert t.found and len(t.calls) == 6 and len(t.crops) == 6
    x0, y0, x1, y1 = t.crops[-1]
    assert t.final_point == pytest.approx([(x0 + x1) / 2, (y0 + y1) / 2])
    assert t.totals()["pixels_sent"] == 6 * 224 * 224


def test_sliding_window_parallel_sends_all_and_takes_first_in_order():
    m = MockClient(['{"absent":true}', '{"point_2d":[0,0]}', '{"point_2d":[1000,1000]}'] + ['{"absent":true}'] * 9)
    t = run("sliding_window", frame(), "cup", m, cfg_with(sliding_window__parallel=True, sliding_window__tile_mode="point"))
    assert len(t.calls) == 12 and t.found
    assert t.final_point == pytest.approx([168.0, 0.0])        # tile 2 (x0=168), its top-left


def test_sliding_window_not_found_sends_every_tile():
    t = run("sliding_window", frame(), "cup", MockClient(), PT)
    assert not t.found and len(t.calls) == 12
    t = run("sliding_window", frame(), "cup", MockClient(default='{"y":false}'))
    assert not t.found and len(t.calls) == 12 and all(c.role == "tile" for c in t.calls)


def test_sliding_window_confirm_mode_localizes_only_yes_tile():
    m = MockClient(['{"y":false}'] * 3 + ['{"y":true}', '{"point_2d":[0,0]}'])
    t = run("sliding_window", frame(), "cup", m)
    assert t.found and [c.role for c in t.calls] == ["tile"] * 4 + ["point"]
    assert t.calls[-1].region == t.calls[-2].region == list(t.crops[-1])
    assert t.final_point == pytest.approx(t.crops[-1][:2])           # tile's top-left = point (0, 0)
    assert "Is there a cup" in m.requests[0]["text"] and "point_2d" not in m.requests[0]["text"]
    yes_no = m.requests[0]["extra_body"]["structured_outputs"]["json"]["anyOf"][1]
    assert yes_no["required"] == ["y"]                                  # point + label off for tile calls


def test_sliding_window_confirm_mode_yes_but_absent_uses_tile_centre():
    m = MockClient(['{"y":true}', '{"absent":true}'])
    t = run("sliding_window", frame(), "cup", m)
    assert t.found and t.final_point == pytest.approx([112.0, 112.0])


def test_sliding_window_confirm_mode_parallel():
    m = MockClient(['{"y":false}', '{"y":true}', '{"y":true}'] + ['{"y":false}'] * 9 + ['{"point_2d":[500,500]}'])
    t = run("sliding_window", frame(), "cup", m, cfg_with(sliding_window__parallel=True))
    assert len(t.calls) == 13 and t.found and t.calls[-1].region == t.calls[1].region


def test_eye_null_first_point_stops_after_one_call():
    t = run("eye_search", frame(), "cup", MockClient(['{"absent":true}']))
    assert not t.found and len(t.calls) == 1 and t.calls[0].role == "point"


@pytest.mark.parametrize("k", [1, 2, 3])
def test_eye_confirmed_on_fixation_k(k):
    pts = ['{"point_2d":[100,100]}', '{"point_2d":[500,500]}', '{"point_2d":[900,900]}']
    answers = []
    for i in range(k):
        answers.append(pts[i])
        answers.append('{"y":true,"point_2d":[500,500],"l":"cup"}' if i == k - 1 else '{"y":false}')
    m = MockClient(answers)
    t = run("eye_search", frame(), "cup", m, cfg_with(confirm__label=True))
    assert t.found and len(t.calls) == 2 * k and len(t.fixations) == k and t.evidence == "cup"
    x0, y0, x1, y1 = t.crops[-1]
    assert t.final_point == pytest.approx([(x0 + x1) / 2, (y0 + y1) / 2])   # refined point = crop centre
    # thumbnail byte-identical across fixations; checked points listed after the image from fixation 2
    thumbs = [r["images"][0].png for r in m.requests[::2]]
    assert all(b == thumbs[0] for b in thumbs) and len({c.image_sha256 for c in t.calls if c.role == "point"}) == 1
    assert t.thumbnail_sha256 == t.calls[0].image_sha256
    if k >= 2:
        assert "(100,100)" in m.requests[2]["text"] and "Already checked" in m.requests[2]["text"]


def test_eye_repeats_counted_and_max_fixations():
    # fixation 1 at the centre, then the model keeps pointing back into the same crop
    m = MockClient(['{"point_2d":[500,500]}', '{"y":false}', '{"point_2d":[510,490]}', '{"point_2d":[500,520]}'])
    t = run("eye_search", frame(), "cup", m)
    assert not t.found and t.repeats == 2
    assert [c.role for c in t.calls] == ["point", "confirm", "point", "point"]   # 3 point calls max
    assert len(t.crops) == 1


def test_eye_max_fixations_all_no():
    m = MockClient(['{"point_2d":[100,100]}', '{"y":false}', '{"point_2d":[500,500]}', '{"y":false}', '{"point_2d":[900,900]}', '{"y":false}'])
    t = run("eye_search", frame(), "cup", m)
    assert not t.found and len(t.calls) == 6 and len(t.crops) == 3 and t.final_point is None


def test_eye_without_refinement_uses_thumbnail_point():
    cfg = cfg_with(confirm__refine_point=False, confirm__label=False)
    m = MockClient(['{"point_2d":[250,250]}', '{"y":true}'])
    t = run("eye_search", frame(), "cup", m, cfg)
    assert t.found and t.final_point == pytest.approx([160.0, 120.0]) and t.evidence is None
    assert "point_2d" not in m.requests[1]["text"] and '"l"' not in m.requests[1]["text"]  # only enabled fields


def test_eye_malformed_answers_are_recorded_not_fatal():
    m = MockClient(['{"point_2d":[100,100],"point_2d":[1,1]}'])
    t = run("eye_search", frame(), "cup", m)
    assert not t.found and t.calls[0].parse_error and t.totals()["parse_errors"] == 1


def test_eye_small_frame_crop_clamped():
    m = MockClient(['{"point_2d":[500,500]}', '{"y":true,"point_2d":[500,500],"l":"x"}'])
    t = run("eye_search", frame(300, 169), "cup", m)
    assert t.crops[0] == [38, 0, 262, 169] and t.calls[1].sent_h == 169


# ------------------------------------------------------------------ accounting + serialization
def test_accounting_adds_up_and_serializes():
    m = MockClient(['{"point_2d":[100,100]}', '{"y":false}', '{"point_2d":[900,900]}', '{"y":true,"point_2d":[500,500],"l":"cup"}'])
    t = run("eye_search", frame(), "cup", m)
    tot = t.totals()
    assert tot["pixels_sent"] == 2 * 96 * 64 + 2 * 224 * 224
    assert tot["pixels_sampled_frac"] == pytest.approx(tot["pixels_sent"] / (640 * 480))
    assert tot["pixels_unique_frac"] == pytest.approx((96 * 64 + 2 * 224 * 224) / (640 * 480))
    assert tot["vision_tokens"] == sum(c.vision_tokens for c in t.calls) == 2 * 6 + 2 * 49
    assert tot["completion_tokens"] == sum(c.completion_tokens for c in t.calls)
    assert tot["time_to_answer_s"] == pytest.approx(sum(c.total_s for c in t.calls))
    assert tot["png_bytes"] == sum(c.png_bytes for c in t.calls)
    d = json.loads(t.to_json())
    assert d["strategy"] == "eye_search" and len(d["calls"]) == 4 and d["n_calls"] == 4
    assert d["calls"][1]["region"] == t.crops[0] and d["prompts_version"] == CFG["prompts"]["version"]


def test_gt_point_check():
    assert in_any_xywh(15, 15, [[10, 10, 5, 5]]) and in_any_xywh(10, 10, [[10, 10, 5, 5]])
    assert not in_any_xywh(16, 15, [[10, 10, 5, 5]]) and not in_any_xywh(1, 1, [])


def test_eye_retry_prompt_variants():
    answers = ['{"point_2d":[100,100]}', '{"y":false}', '{"absent":true}']
    m = MockClient(list(answers))
    run("eye_search", frame(), "cup", m, cfg_with(retry_prompt="encourage"))
    t2 = m.requests[2]["text"]
    assert "may still be elsewhere" in t2 and "(100,100)" in t2 and "Already checked" not in t2
    m = MockClient(list(answers))
    run("eye_search", frame(), "cup", m, cfg_with(retry_prompt="plain"))
    assert "Already checked" in m.requests[2]["text"]
    assert "Already checked" not in m.requests[0]["text"] and "may still" not in m.requests[0]["text"]


def test_label_off_by_default_and_trimmed_when_on():
    from foveated_search.search.schemas import trim_label

    m = MockClient(['{"point_2d":[500,500]}', '{"y":true,"point_2d":[500,500]}'])
    t = run("eye_search", frame(), "cup", m)
    assert t.found and t.evidence is None and '"l"' not in m.requests[1]["text"]
    assert trim_label("a banana split with a w", 23) == "a banana split with a"
    assert trim_label("red mug", 48) == "red mug"
    assert trim_label("supercalifragilistic", 20) == "supercalifragilistic"   # no space: keep
    assert trim_label("  ", 48) is None and trim_label(None, 48) is None
