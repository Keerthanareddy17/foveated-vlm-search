"""Step 8b (GPU-free): mosaic manifests (geometry, filler rules, regeneration), masked thumbnails, splits,
and the latency-mode rule."""

import asyncio
import json

import numpy as np
import pytest
from PIL import Image

from foveated_search.config import REPO_ROOT
from foveated_search.data.coco import coco_dir
from foveated_search.data.manifest import read_jsonl
from foveated_search.data.mosaic import manifest_path, mosaic_config
from foveated_search.eval.runner import run_mode
from foveated_search.eval.splits import load_split

CFG = mosaic_config()
M = CFG["mosaic"]
ROWS = {g: read_jsonl(manifest_path(CFG, g)) for g in M["grids"]}
ANN = coco_dir(CFG) / "instances_val2017.json"
needs_ann = pytest.mark.skipif(not ANN.exists(), reason="COCO val annotations not on the local disk")


@pytest.mark.parametrize("g", [2, 3])
def test_boxes_land_in_the_query_cell(g):
    src = {r["query_id"]: r for r in load_split(CFG, "test")}
    for r in ROWS[g]:
        assert (r["width"], r["height"]) == (g * M["cell_w"], g * M["cell_h"]) and len(r["cells"]) == g * g
        qc = r["cells"][r["query_cell"]]
        assert qc["image_id"] == r["image_id"] and (qc["w"], qc["h"]) == (src[r["query_id"]]["width"], src[r["query_id"]]["height"])
        cx, cy = (r["query_cell"] % g) * M["cell_w"], (r["query_cell"] // g) * M["cell_h"]
        assert cx <= qc["x"] and qc["x"] + qc["w"] <= cx + M["cell_w"] and cy <= qc["y"] and qc["y"] + qc["h"] <= cy + M["cell_h"]
        assert len(r["gt_boxes"]) == len(src[r["query_id"]]["gt_boxes"])
        for (x, y, w, h), (ox, oy, ow, oh) in zip(r["gt_boxes"], src[r["query_id"]]["gt_boxes"]):
            assert (w, h) == (ow, oh) and x == pytest.approx(ox + qc["x"]) and y == pytest.approx(oy + qc["y"])
            assert qc["x"] <= x and x + w <= qc["x"] + qc["w"] + 0.01 and qc["y"] <= y and y + h <= qc["y"] + qc["h"] + 0.01
        assert len({c["image_id"] for c in r["cells"]}) == g * g          # no image twice in a mosaic


def test_splits_and_sizes():
    assert len(load_split(CFG, "test_fit")) == len(ROWS[2]) == len(ROWS[3]) == 308
    assert {r["query_id"] for r in ROWS[2]} == {r["query_id"] for r in load_split(CFG, "test_fit")}
    lat = {r["query_id"] for r in load_split(CFG, "latency_fit")}
    assert lat == {r["query_id"] for r in load_split(CFG, "latency_m3")} and len(lat) == 61


def test_latency_mode_rule():
    assert run_mode("latency") == run_mode("latency_fit") == run_mode("latency_m3") == "latency"
    assert run_mode("test") == run_mode("test_m2") == run_mode("all") == "accuracy"


@needs_ann
@pytest.mark.parametrize("g", [2, 3])
def test_fillers_never_contain_the_target_supercategory(g):
    from foveated_search.data.coco import CocoIndex

    idx = CocoIndex.load(ANN)
    cats = {c["name"]: c for c in idx.cats.values()}
    food = {c["id"] for c in idx.cats.values() if c["supercategory"] == "food"}
    excl = {cats[n]["id"] for n in CFG.get_key("train_data.food_negative_exclude")}
    for r in ROWS[g]:
        blocked = food | excl if r["query"] == "food" else \
            {c["id"] for c in idx.cats.values() if c["supercategory"] == cats[r["query"]]["supercategory"]}
        for c in r["cells"]:
            if c["cell"] != r["query_cell"]:
                present = {a["category_id"] for a in idx.anns_by_image.get(c["image_id"], [])}
                assert not present & blocked, (r["query_id"], c["image_id"])


@needs_ann
def test_regeneration_identical():
    from foveated_search.data.coco import CocoIndex
    from foveated_search.data.mosaic import build_manifest

    idx = CocoIndex.load(ANN)
    for g in M["grids"]:
        data = "".join(json.dumps(r, separators=(",", ":")) + "\n" for r in build_manifest(CFG, g, idx)).encode()
        assert data == manifest_path(CFG, g).read_bytes()


def test_rendered_mosaic_has_the_query_image_unscaled():
    from foveated_search.data.fetch_images import eval_image_path
    from foveated_search.data.mosaic import render

    r = ROWS[3][0]
    img = np.asarray(render(CFG, r))
    qc = r["cells"][r["query_cell"]]
    src = np.asarray(Image.open(eval_image_path(CFG, qc["file_name"])).convert("RGB"))
    assert np.array_equal(img[qc["y"]:qc["y"] + qc["h"], qc["x"]:qc["x"] + qc["w"]], src)
    if qc["x"] % M["cell_w"]:                                     # gray padding left of a narrower image
        assert tuple(img[qc["y"], qc["x"] - 1]) == tuple(M["pad_rgb"])


def test_masked_thumbnail_and_eye_search_option():
    from foveated_search.search.base import run_strategy
    from foveated_search.search.strategies import masked_thumbnail
    from tests.mock_client import MockClient
    from tests.test_search import cfg_with, frame

    thumb = Image.new("RGB", (96, 64), (255, 0, 0))
    m = np.asarray(masked_thumbnail(thumb, [[0, 0, 320, 240]], (640, 480), (128, 128, 128)))
    assert tuple(m[0, 0]) == (128, 128, 128) and tuple(m[31, 47]) == (128, 128, 128) and tuple(m[40, 60]) == (255, 0, 0)
    answers = ['{"point_2d":[100,100]}', '{"y":false}', '{"point_2d":[900,900]}', '{"y":true,"point_2d":[500,500]}']
    mc = MockClient(list(answers))
    t = asyncio.run(run_strategy("eye_search", frame(), "cup", mc, cfg_with(look_again_mask=True)))
    assert t.found and t.params["look_again_mask"] is True
    thumbs = [r["images"][0].png for r in mc.requests[::2]]
    assert thumbs[0] != thumbs[1]                                 # fixation 2 sees the checked crop grayed out
    assert "(100,100)" in mc.requests[2]["text"]                  # and the text still lists the checked point
    mc = MockClient(list(answers))
    asyncio.run(run_strategy("eye_search", frame(), "cup", mc, cfg_with(look_again_mask=False)))
    assert mc.requests[0]["images"][0].png == mc.requests[2]["images"][0].png   # default: byte-identical
