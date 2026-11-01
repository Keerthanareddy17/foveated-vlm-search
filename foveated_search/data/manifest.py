"""Build the frozen evaluation manifest (Step 3). Deterministic: same seed + annotations -> same bytes.

  python -m foveated_search.data.manifest all      # annotations -> manifest -> images -> stats -> sheet
  python -m foveated_search.data.manifest build    # manifest + dev subset + image ids only

Query rules (parameters in configs/eval.yaml):
- One query per image. Query = a COCO category name or the "food" supercategory.
- Positive: the target has >= 1 non-crowd instance; GT = all non-crowd boxes of the target (for
  "food": all boxes of all food categories). The LARGEST instance box must be >= min_box_area_frac
  of the image. Optionally skip if the target also has a crowd annotation (crowd isn't GT).
- Negative: no annotation at all (crowd included) of ANY category in the target's supercategory
  (for "food": no food category at all). Guards against COCO's incomplete labels.
- Food queries get `food_share` of positives and of negatives; single-category queries are drawn
  round-robin over all 80 categories, capped at max_per_category.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

from foveated_search.config import REPO_ROOT, Config, load_config
from foveated_search.data.coco import CocoIndex

FIELDS = ["query_id", "image_id", "file_name", "coco_url", "width", "height", "query", "query_type",
          "is_negative", "gt_boxes", "gt_category_names", "num_instances", "size_bucket",
          "max_box_area_frac"]


def eval_config(overlays: list[str] | None = None) -> Config:
    return load_config(overlays=["configs/eval.yaml", *(overlays or [])])


def _size_bucket(area: float, e: dict) -> str:
    if area < e["size_small_max"]:
        return "small"
    if area < e["size_medium_max"]:
        return "medium"
    return "large"


class Builder:
    def __init__(self, idx: CocoIndex, cfg: Config):
        self.idx, self.cfg, self.e = idx, cfg, cfg["eval"]
        self.food_cats = idx.supercat_members(self.e["food_supercategory"])
        if not self.food_cats:
            raise ValueError(f"no categories in supercategory {self.e['food_supercategory']!r}")
        self.cat_names = sorted(c["name"] for c in idx.cats.values())
        self.cat_by_name = {c["name"]: c for c in idx.cats.values()}
        # images with at least one non-crowd annotation (real, labelled scenes), sorted for determinism
        self.image_ids = sorted(i for i, anns in idx.anns_by_image.items() if any(not a["iscrowd"] for a in anns))

    # ------------------------------------------------------------------ rules
    def target_cats(self, query: str) -> set[int]:
        return set(self.food_cats) if query == self.e["food_supercategory"] else {self.cat_by_name[query]["id"]}

    def blocked_cats(self, query: str) -> set[int]:
        """Categories that must be fully absent for a negative: the target's whole supercategory."""
        if query == self.e["food_supercategory"]:
            return set(self.food_cats)
        return self.idx.supercat_members(self.cat_by_name[query]["supercategory"])

    def positive(self, image_id: int, query: str) -> dict | None:
        im = self.idx.images[image_id]
        tc = self.target_cats(query)
        anns = [a for a in self.idx.anns_by_image.get(image_id, []) if a["category_id"] in tc]
        if self.e["skip_positive_if_crowd"] and any(a["iscrowd"] for a in anns):
            return None
        inst = [a for a in anns if not a["iscrowd"]]
        if not inst:
            return None
        img_area = im["width"] * im["height"]
        largest = max(inst, key=lambda a: (a["bbox"][2] * a["bbox"][3], -a["id"]))
        frac = largest["bbox"][2] * largest["bbox"][3] / img_area
        if frac < self.e["min_box_area_frac"]:
            return None
        nd = int(self.e["coord_decimals"])
        return {
            "gt_boxes": [[round(v, nd) for v in a["bbox"]] for a in inst],
            "gt_category_names": [self.idx.cats[a["category_id"]]["name"] for a in inst],
            "num_instances": len(inst),
            "size_bucket": _size_bucket(largest["area"], self.e),
            "max_box_area_frac": round(frac, 5),
        }

    def negative_ok(self, image_id: int, query: str) -> bool:
        blocked = self.blocked_cats(query)
        return not any(a["category_id"] in blocked for a in self.idx.anns_by_image.get(image_id, []))

    # ------------------------------------------------------------------ selection
    def build(self) -> list[dict]:
        e = self.e
        rng = random.Random(int(self.cfg["seed"]))
        n = int(e["num_queries"])
        n_neg = round(n * float(e["negatives_share"]))
        n_pos = n - n_neg
        n_food_pos = round(n_pos * float(e["food_share"]))
        n_food_neg = round(n_neg * float(e["food_share"]))
        food = e["food_supercategory"]
        cap = int(e["max_per_category"])

        order = list(self.image_ids)
        rng.shuffle(order)
        used: set[int] = set()
        rows: list[dict] = []

        def take(query: str, negative: bool, count: int) -> int:
            got = 0
            for iid in order:
                if got == count:
                    break
                if iid in used:
                    continue
                info = None if negative else self.positive(iid, query)
                if (negative and self.negative_ok(iid, query)) or (not negative and info):
                    rows.append(self._row(iid, query, negative, info))
                    used.add(iid)
                    got += 1
            return got

        def round_robin(negative: bool, total: int) -> None:
            cats = list(self.cat_names)
            rng.shuffle(cats)
            counts: Counter = Counter()
            exhausted: set[str] = set()
            got = 0
            while got < total:
                progressed = False
                for c in cats:
                    if got == total:
                        break
                    if c in exhausted or counts[c] >= cap:
                        continue
                    if take(c, negative, 1):
                        counts[c] += 1
                        got += 1
                        progressed = True
                    else:
                        exhausted.add(c)
                if not progressed:
                    raise RuntimeError(f"cannot fill {total} {'negative' if negative else 'positive'} "
                                       f"category queries (got {got}); raise max_per_category")

        for label, negative, count in [("food positives", False, n_food_pos), ("food negatives", True, n_food_neg)]:
            got = take(food, negative, count)
            if got != count:
                raise RuntimeError(f"only {got}/{count} {label} available")
        round_robin(False, n_pos - n_food_pos)
        round_robin(True, n_neg - n_food_neg)

        rng.shuffle(rows)  # mix query types in file order
        for i, r in enumerate(rows):
            r["query_id"] = f"q{i:04d}"
        return [{k: r[k] for k in FIELDS} for r in rows]

    def _row(self, iid: int, query: str, negative: bool, info: dict | None) -> dict:
        im = self.idx.images[iid]
        r = {"image_id": iid, "file_name": im["file_name"], "coco_url": im["coco_url"],
             "width": im["width"], "height": im["height"], "query": query,
             "query_type": "supercategory" if query == self.e["food_supercategory"] else "category",
             "is_negative": negative}
        if negative:
            r.update(gt_boxes=[], gt_category_names=[], num_instances=0, size_bucket=None, max_box_area_frac=0.0)
        else:
            r.update(info)
        return r


def dev_subset(rows: list[dict], size: int, seed: int) -> list[dict]:
    """Stratified by (query_type, is_negative), proportional, largest-remainder; keeps manifest order."""
    strata: dict[tuple, list[dict]] = {}
    for r in rows:
        strata.setdefault((r["query_type"], r["is_negative"]), []).append(r)
    keys = sorted(strata)
    exact = {k: size * len(strata[k]) / len(rows) for k in keys}
    alloc = {k: int(exact[k]) for k in keys}
    for k in sorted(keys, key=lambda k: (-(exact[k] - alloc[k]), k))[: size - sum(alloc.values())]:
        alloc[k] += 1
    rng = random.Random(seed)
    chosen = {r["query_id"] for k in keys for r in rng.sample(strata[k], alloc[k])}
    return [r for r in rows if r["query_id"] in chosen]


def to_jsonl(rows: list[dict]) -> bytes:
    return "".join(json.dumps(r, separators=(",", ":"), ensure_ascii=False) + "\n" for r in rows).encode()


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def build_outputs(cfg: Config, idx: CocoIndex) -> dict[str, str]:
    """Build manifest + dev subset + image id list. Returns {path: sha256}."""
    rows = Builder(idx, cfg).build()
    dev = dev_subset(rows, int(cfg.get_key("eval.dev_size")), int(cfg["seed"]))
    o = cfg["outputs"]
    files = {
        o["manifest"]: to_jsonl(rows),
        o["dev_manifest"]: to_jsonl(dev),
        o["image_ids"]: "".join(f"{i}\n" for i in sorted(r["image_id"] for r in rows)).encode(),
    }
    hashes = {}
    for rel, data in files.items():
        (REPO_ROOT / rel).write_bytes(data)
        hashes[rel] = sha256(data)
        print(f"[manifest] wrote {rel}  sha256 {hashes[rel]}")
    return hashes


def read_jsonl(path: str | Path) -> list[dict]:
    with open(REPO_ROOT / path) as f:
        return [json.loads(line) for line in f]


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    cfg = eval_config()
    from foveated_search.data.coco import ensure_annotations

    paths = ensure_annotations(cfg)
    idx = CocoIndex.load(paths["instances_val2017.json"])
    print(f"[coco] val2017: {len(idx.images)} images, {sum(len(v) for v in idx.anns_by_image.values())} annotations, "
          f"{len(idx.cats)} categories")
    if cmd in ("build", "all"):
        build_outputs(cfg, idx)
    if cmd in ("fetch", "all"):
        from foveated_search.data.fetch_images import fetch_eval_images

        fetch_eval_images(cfg, read_jsonl(cfg.get_key("outputs.manifest")))
    if cmd in ("stats", "all"):
        from foveated_search.data.manifest_report import contact_sheet, write_stats

        rows = read_jsonl(cfg.get_key("outputs.manifest"))
        write_stats(cfg, rows, read_jsonl(cfg.get_key("outputs.dev_manifest")))
        contact_sheet(cfg, rows)


if __name__ == "__main__":
    main()
