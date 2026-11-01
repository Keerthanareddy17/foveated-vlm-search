"""Eval manifest rules (Step 3). GPU-free.

- Checks on the committed manifest always run.
- Checks against COCO annotations (and the regenerate-same-bytes check) need the local
  instances_val2017.json; they are skipped if `make eval-manifest` hasn't run this session.
- A tiny synthetic COCO tests the builder rules without any data.
"""

import json
from collections import Counter

import pytest

from foveated_search.config import REPO_ROOT
from foveated_search.data.coco import CocoIndex, coco_dir
from foveated_search.data.manifest import FIELDS, Builder, dev_subset, eval_config, read_jsonl, to_jsonl

CFG = eval_config()
E = CFG["eval"]
FOOD = E["food_supercategory"]
ROWS = read_jsonl(CFG.get_key("outputs.manifest"))
DEV = read_jsonl(CFG.get_key("outputs.dev_manifest"))
ANN = coco_dir(CFG) / "instances_val2017.json"
needs_ann = pytest.mark.skipif(not ANN.exists(), reason="COCO annotations not on local disk (run make eval-manifest)")


@pytest.fixture(scope="module")
def idx():
    return CocoIndex.load(ANN)


# ------------------------------------------------------------------ committed manifest
def test_shape_and_mix():
    assert len(ROWS) == E["num_queries"]
    assert all(list(r) == FIELDS for r in ROWS)
    assert len({r["query_id"] for r in ROWS}) == len(ROWS)
    neg = sum(r["is_negative"] for r in ROWS)
    assert neg == round(E["num_queries"] * E["negatives_share"])
    pos = [r for r in ROWS if not r["is_negative"]]
    assert sum(r["query"] == FOOD for r in pos) == round(len(pos) * E["food_share"])
    assert all(r["query_type"] == ("supercategory" if r["query"] == FOOD else "category") for r in ROWS)


def test_no_image_twice():
    ids = [r["image_id"] for r in ROWS]
    assert len(ids) == len(set(ids))
    listed = (REPO_ROOT / CFG.get_key("outputs.image_ids")).read_text().split()
    assert sorted(map(int, listed)) == sorted(ids)


def test_negatives_have_no_boxes():
    for r in ROWS:
        if r["is_negative"]:
            assert r["gt_boxes"] == [] and r["num_instances"] == 0 and r["size_bucket"] is None
        else:
            assert len(r["gt_boxes"]) == r["num_instances"] == len(r["gt_category_names"]) > 0


def test_boxes_inside_image():
    for r in ROWS:
        for x, y, w, h in r["gt_boxes"]:
            assert w > 0 and h > 0
            assert -0.01 <= x and -0.01 <= y and x + w <= r["width"] + 0.01 and y + h <= r["height"] + 0.01


def test_min_area_rule():
    for r in ROWS:
        if r["is_negative"]:
            continue
        largest = max(w * h for _, _, w, h in r["gt_boxes"]) / (r["width"] * r["height"])
        assert largest >= E["min_box_area_frac"] - 1e-4  # boxes are rounded to 2 decimals
        assert r["max_box_area_frac"] >= E["min_box_area_frac"]


def test_category_cap():
    c = Counter(r["query"] for r in ROWS if r["query_type"] == "category" and not r["is_negative"])
    assert max(c.values()) <= E["max_per_category"]


def test_dev_subset_is_stratified_sample():
    assert len(DEV) == E["dev_size"]
    main = {r["query_id"]: r for r in ROWS}
    assert all(main[r["query_id"]] == r for r in DEV)
    strata = lambda rows: Counter((r["query_type"], r["is_negative"]) for r in rows)  # noqa: E731
    for k, v in strata(ROWS).items():
        assert abs(strata(DEV)[k] - v * len(DEV) / len(ROWS)) <= 1


# ------------------------------------------------------------------ against annotations
@needs_ann
def test_negatives_have_no_same_supercategory(idx):
    b = Builder(idx, CFG)
    for r in ROWS:
        if r["is_negative"]:
            present = {a["category_id"] for a in idx.anns_by_image[r["image_id"]]}  # crowd included
            assert not present & b.blocked_cats(r["query"]), r["query_id"]


@needs_ann
def test_gt_is_all_noncrowd_target_boxes(idx):
    b = Builder(idx, CFG)
    food_names = {idx.cats[c]["name"] for c in b.food_cats}
    for r in ROWS:
        if r["is_negative"]:
            continue
        if r["query"] == FOOD:
            assert set(r["gt_category_names"]) <= food_names  # food queries only use food boxes
        else:
            assert set(r["gt_category_names"]) == {r["query"]}
        anns = [a for a in idx.anns_by_image[r["image_id"]] if a["category_id"] in b.target_cats(r["query"])]
        assert not any(a["iscrowd"] for a in anns)  # skip_positive_if_crowd
        assert r["num_instances"] == len(anns)
        im = idx.images[r["image_id"]]
        assert (r["width"], r["height"], r["file_name"], r["coco_url"]) == (im["width"], im["height"], im["file_name"], im["coco_url"])


@needs_ann
def test_regenerate_same_bytes(idx):
    rows = Builder(idx, CFG).build()
    assert to_jsonl(rows) == (REPO_ROOT / CFG.get_key("outputs.manifest")).read_bytes()
    dev = dev_subset(rows, E["dev_size"], CFG["seed"])
    assert to_jsonl(dev) == (REPO_ROOT / CFG.get_key("outputs.dev_manifest")).read_bytes()


# ------------------------------------------------------------------ synthetic COCO (no data needed)
def _tiny_coco():
    cats = [{"id": 1, "name": "banana", "supercategory": "food"}, {"id": 2, "name": "pizza", "supercategory": "food"},
            {"id": 3, "name": "cup", "supercategory": "kitchen"}, {"id": 4, "name": "bowl", "supercategory": "kitchen"},
            {"id": 5, "name": "dog", "supercategory": "animal"}]
    images, anns, aid = [], [], 0

    def ann(iid, cid, box, crowd=0):
        nonlocal aid
        aid += 1
        anns.append({"id": aid, "image_id": iid, "category_id": cid, "bbox": box, "area": box[2] * box[3], "iscrowd": crowd})

    for i in range(1, 41):
        images.append({"id": i, "file_name": f"{i:012d}.jpg", "coco_url": f"http://x/{i}.jpg", "width": 100, "height": 100})
        ann(i, 5, [0, 0, 30, 30])                      # every image has a dog
        if i <= 10:
            ann(i, 1 + i % 2, [10, 10, 40, 40])        # food, big
        elif i <= 20:
            ann(i, 3, [50, 50, 30, 30])                # cup, big
        elif i <= 25:
            ann(i, 3, [50, 50, 5, 5])                  # cup, tiny (0.25% < 0.5%)
        elif i <= 28:
            ann(i, 4, [50, 50, 30, 30]); ann(i, 4, [0, 50, 40, 40], crowd=1)  # bowl with crowd  # noqa: E702
    return CocoIndex.from_dict({"images": images, "categories": cats, "annotations": anns})


def test_builder_rules_on_tiny_coco():
    idx = _tiny_coco()
    cfg = eval_config()
    b = Builder(idx, cfg)
    assert b.positive(1, "food") and b.positive(11, "cup")
    assert b.positive(21, "cup") is None                       # below min_box_area_frac
    assert b.positive(26, "bowl") is None                      # target has a crowd annotation
    assert b.positive(1, "cup") is None                        # absent
    assert b.negative_ok(30, "cup") and b.negative_ok(30, "food")
    assert not b.negative_ok(26, "cup")                        # bowl present (same supercategory), even crowd
    assert not b.negative_ok(1, "banana") and not b.negative_ok(2, "food")
    pos = b.positive(2, "food")
    assert pos["gt_category_names"] == ["banana"] and pos["size_bucket"] == "medium"  # 40x40=1600 < 96^2
    assert pos["max_box_area_frac"] == pytest.approx(0.16)


def test_builder_is_deterministic_on_tiny_coco():
    idx = _tiny_coco()
    cfg = eval_config()
    cfg["eval"].update(num_queries=10, negatives_share=0.2, food_share=0.25, max_per_category=3)
    a, b = Builder(idx, cfg).build(), Builder(idx, cfg).build()
    assert to_jsonl(a) == to_jsonl(b)
    assert len({r["image_id"] for r in a}) == 10
    for r in a:
        if r["is_negative"]:
            assert Builder(idx, cfg).negative_ok(r["image_id"], r["query"])
    assert json.loads(to_jsonl(a).splitlines()[0])["query_id"] == "q0000"
