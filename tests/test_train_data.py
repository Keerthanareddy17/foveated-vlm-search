"""Step 6 training data (GPU-free).

Unit tests (always run): target serializers round-trip through the parsers, on-object points lie in
the mask, "food" negative rule, eye search and data generation build identical prompt text.
Dataset tests (need the generated data on the local disk; skipped otherwise): checked points outside
target boxes, food negatives contain no excluded class, train/val/eval disjoint, PNG sizes as recorded,
and seeded regeneration of a sample subset is byte-identical.
"""

import json
import random

import numpy as np
import pytest

from foveated_search.config import REPO_ROOT
from foveated_search.data.coco import CocoIndex, coco_dir
from foveated_search.data.masks import on_object_point, point_in_mask
from foveated_search.search.base import confirm_prompt, point_prompt
from foveated_search.search.schemas import dump_confirm, dump_point, parse_confirm, parse_p

# ------------------------------------------------------------------ serializers
@pytest.mark.parametrize("p", [(0, 0), (1000, 1000), (123.4, 567.6), (500, 1)])
def test_point_target_roundtrip(p):
    s = dump_point(p)
    assert parse_p(s) == (round(p[0]), round(p[1]))
    assert s == json.dumps({"point_2d": [round(p[0]), round(p[1])]})          # xgrammar-style separators
    assert ": " in s and ", " in s


def test_absent_and_confirm_roundtrip():
    assert dump_point(None) == '{"absent": true}' and parse_p(dump_point(None)) is None
    assert dump_confirm(False) == '{"y": false}' and parse_confirm(dump_confirm(False)).yes is False
    a = parse_confirm(dump_confirm(True, (12, 988), "hot dog"))
    assert (a.yes, a.point, a.label) == (True, (12.0, 988.0), "hot dog")
    assert dump_confirm(True, (5, 6)) == '{"y": true, "point_2d": [5, 6]}'
    with pytest.raises(ValueError):
        dump_point((1001, 3))


# ------------------------------------------------------------------ on-object point
def _shapes():
    yy, xx = np.mgrid[0:120, 0:120]
    r2 = (xx - 60) ** 2 + (yy - 60) ** 2
    donut = (r2 <= 50 ** 2) & (r2 >= 32 ** 2)
    ell = np.zeros((80, 80), bool)
    ell[5:75, 5:14] = True
    ell[66:75, 5:75] = True
    thin = np.zeros((50, 200), bool)
    thin[24:26, 10:190] = True
    arc = (r2 <= 40 ** 2) & (r2 >= 36 ** 2) & (yy > 60)
    disk = r2 <= 10 ** 2
    return {"donut": donut, "L": ell, "thin": thin, "arc": arc, "disk": disk}


@pytest.mark.parametrize("name", ["donut", "L", "thin", "arc", "disk"])
def test_on_object_point_inside_mask(name):
    m = _shapes()[name]
    p = on_object_point(m)
    assert p is not None and point_in_mask(m, *p)
    if name == "disk":
        assert p == (60.5, 60.5)          # centroid inside -> centroid
    if name == "donut":
        assert 32 <= ((p[0] - 60) ** 2 + (p[1] - 60) ** 2) ** 0.5 <= 50   # on the ring, not the empty centre


def test_on_object_point_empty():
    assert on_object_point(np.zeros((5, 5), bool)) is None


# ------------------------------------------------------------------ prompt consistency with eye search
def test_prompts_identical_to_eye_search():
    import asyncio

    from foveated_search.search.base import run_strategy, search_config
    from tests.mock_client import MockClient
    from tests.test_search import frame

    cfg = search_config()
    m = MockClient(['{"point_2d":[100,100]}', '{"y":false}', '{"absent":true}'])
    asyncio.run(run_strategy("eye_search", frame(), "cup", m, cfg))
    assert m.requests[0]["text"] == point_prompt(cfg, "cup")
    assert m.requests[1]["text"] == confirm_prompt(cfg, "cup", refine_point=True, label=False)
    assert m.requests[2]["text"] == point_prompt(cfg, "cup", [(100, 100)])


# ------------------------------------------------------------------ rules on a tiny synthetic COCO
def _tiny():
    from foveated_search.data.train_data import Rules, data_config

    cats = [{"id": 1, "name": "pizza", "supercategory": "food"}, {"id": 2, "name": "cup", "supercategory": "kitchen"},
            {"id": 3, "name": "dining table", "supercategory": "furniture"}, {"id": 4, "name": "dog", "supercategory": "animal"},
            {"id": 5, "name": "bowl", "supercategory": "kitchen"}]
    imgs = [{"id": i, "file_name": f"{i}.jpg", "coco_url": "", "width": 100, "height": 100} for i in range(1, 5)]
    anns = [{"id": 1, "image_id": 1, "category_id": 4, "bbox": [0, 0, 10, 10], "area": 100, "iscrowd": 0},
            {"id": 2, "image_id": 2, "category_id": 3, "bbox": [0, 0, 10, 10], "area": 100, "iscrowd": 0},
            {"id": 3, "image_id": 3, "category_id": 1, "bbox": [0, 0, 10, 10], "area": 100, "iscrowd": 0},
            {"id": 4, "image_id": 4, "category_id": 5, "bbox": [0, 0, 10, 10], "area": 100, "iscrowd": 0}]
    idx = CocoIndex.from_dict({"images": imgs, "categories": cats, "annotations": anns})
    cfg = data_config()
    cfg["train_data"]["food_negative_exclude"] = ["dining table", "cup", "bowl"]
    return Rules(idx, cfg)


def test_food_negative_rule_excludes_kitchen_scenes():
    R = _tiny()
    assert R.negative_ok(1, "food")              # dog only
    assert not R.negative_ok(2, "food")          # dining table: might hold unlabelled food
    assert not R.negative_ok(3, "food")          # pizza
    assert not R.negative_ok(4, "food")          # bowl
    assert not R.negative_ok(4, "cup")           # bowl = same supercategory as cup
    assert R.negative_ok(1, "cup") and R.no_crop_image_ok(2, "cup") and not R.no_crop_image_ok(2, "food")


# ------------------------------------------------------------------ generated dataset
def _cfg():
    from foveated_search.data.train_data import data_config

    return data_config()


def _data():
    from foveated_search.data.train_data import out_dir

    od = out_dir(_cfg())
    if not (od / "train.jsonl").exists():
        return None
    load = lambda p: [json.loads(line) for line in open(p)]  # noqa: E731
    return od, load(od / "train.jsonl"), load(od / "val.jsonl")


DATA = _data()
needs_data = pytest.mark.skipif(DATA is None, reason="training data not generated (make gen-data)")


@needs_data
def test_every_target_parses_and_matches_type():
    _, tr, va = DATA
    for s in tr + va:
        t = s["messages"][1]["content"][0]["text"]
        if s["type"] == "confirm":
            a = parse_confirm(t)
            assert a.yes == (s["meta"]["polarity"] == "yes")
            assert (a.label is not None) == (a.yes and s["meta"]["label_on"])
        else:
            p = parse_p(t)
            assert (p is None) == (s["meta"]["polarity"] == "neg")
        assert s["messages"][0]["content"][0]["type"] == "image"            # image first, then text


@needs_data
def test_checked_points_outside_all_target_boxes():
    _, tr, va = DATA
    n = 0
    for s in tr + va:
        if s["type"] != "look_again":
            continue
        m = s["meta"]
        assert m["checked_model"] and len(m["checked_model"]) <= 2
        for (cx, cy), (fx, fy) in zip(m["checked_model"], m["checked_frame"]):
            for x, y, w, h in m["gt_boxes_frame"] + m.get("avoid_boxes_frame", []):   # food: also food-adjacent boxes
                assert not (x <= fx <= x + w and y <= fy <= y + h), s["id"]
            assert f"({cx},{cy})" in s["messages"][0]["content"][1]["text"]
            n += 1
    assert n > 0


@needs_data
def test_food_negatives_contain_no_excluded_class():
    from foveated_search.data.train_data import data_config

    cfg = data_config()
    idx = CocoIndex.load(coco_dir(cfg) / cfg.get_key("train_data.source_annotations"))
    excl = set(cfg.get_key("train_data.food_negative_exclude"))
    food = {c["name"] for c in idx.cats.values() if c["supercategory"] == "food"}
    _, tr, va = DATA
    n = 0
    for s in tr + va:
        if s["meta"]["query"] == "food" and s["meta"]["polarity"] in ("neg", "no"):
            names = {idx.cats[a["category_id"]]["name"] for a in idx.anns_by_image.get(s["meta"]["image_id"], [])}
            assert not names & (excl | food), (s["id"], names & (excl | food))
            n += 1
    assert n > 100


@needs_data
def test_splits_disjoint_and_no_eval_overlap():
    _, tr, va = DATA
    tri, vai = {s["meta"]["image_id"] for s in tr}, {s["meta"]["image_id"] for s in va}
    ev = {int(x) for x in (REPO_ROOT / "results/eval_image_ids.txt").read_text().split()}
    assert not tri & vai and not (tri | vai) & ev
    assert all(s["split"] == "train" for s in tr) and all(s["split"] == "val" for s in va)


@needs_data
def test_seeded_regeneration_identical():
    """Re-run the generator for a sample of images in-process: identical records and PNG bytes."""
    from foveated_search.data.train_data import _init_worker, _run_image, data_config, plan

    cfg = data_config()
    idx = CocoIndex.load(coco_dir(cfg) / cfg.get_key("train_data.source_annotations"))
    p = plan(idx, cfg)
    od, tr, va = DATA
    stored = {s["id"]: s for s in tr + va}
    assert len(p["jobs"]) == sum(1 for _ in p["jobs"])
    by_img = {}
    for j in p["jobs"]:
        by_img.setdefault(j["image_id"], []).append(j)
    rng = random.Random(0)
    _init_worker(dict(cfg), idx)
    for iid in rng.sample(sorted(by_img), 15):
        before = {j["id"]: (od / f"images/{j['id']}.png").read_bytes() for j in by_img[iid] if j["id"] in stored}
        for sid, rec in _run_image((iid, by_img[iid])):
            if rec is None:
                assert sid not in stored
                continue
            assert json.dumps(rec, sort_keys=True) == json.dumps(stored[sid], sort_keys=True)
            assert (od / rec["image"]).read_bytes() == before[sid]


@needs_data
def test_target_points_on_target_masks():
    """Positive / yes targets: the 0..1000 point, mapped back to frame pixels, lies on a target mask
    (within 1 px: the model's integer coordinates are coarser than pixels on big frames)."""
    from foveated_search.data.masks import seg_to_mask
    from foveated_search.data.train_data import Rules, data_config

    cfg = data_config()
    idx = CocoIndex.load(coco_dir(cfg) / cfg.get_key("train_data.source_annotations"))
    R = Rules(idx, cfg)
    _, tr, va = DATA
    pos = [s for s in tr + va if s["meta"]["polarity"] in ("pos", "yes")]
    for s in random.Random(1).sample(pos, 300):
        m = s["meta"]
        im = idx.images[m["image_id"]]
        H, W = im["height"], im["width"]
        union = np.zeros((H, W), bool)
        for a in R.targets(m["image_id"], m["query"]):
            union |= seg_to_mask(a["segmentation"], H, W)
        x0, y0, x1, y1 = m["region"]
        fx = x0 + m["target_model"][0] / 1000 * (x1 - x0)
        fy = y0 + m["target_model"][1] / 1000 * (y1 - y0)
        near = union[max(0, int(fy) - 1):int(fy) + 2, max(0, int(fx) - 1):int(fx) + 2]
        assert near.any(), (s["id"], fx, fy)
