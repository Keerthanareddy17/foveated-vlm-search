"""Step 6: generate training data for the two search skills, seeded and train/inference-consistent.

  python -m foveated_search.data.train_data all     # plan -> fetch images -> generate -> stats + grid
  python -m foveated_search.data.train_data plan|fetch|generate|report

Sample types (configs/train_data.yaml):
  where       (Type 1)  thumbnail + point prompt                      -> {"point_2d": [x, y]} | {"absent": true}
  look_again  (Type 1b) thumbnail + point prompt + already-checked    -> same targets
  confirm     (Type 2)  sharp crop + confirm prompt (refined point on) -> {"y": false} | {"y": true, "point_2d": [...]}
Consistency with eye search: thumbnails/crops from search.views, prompt text from search.base
(point_prompt / confirm_prompt), targets from search.schemas serializers, PNG bytes from the same
encoder the client uses. Each sample's randomness comes from its own id, so regeneration is identical.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

from PIL import Image

from foveated_search.config import REPO_ROOT, Config
from foveated_search.data.coco import CocoIndex, coco_dir
from foveated_search.data.masks import on_object_point, seg_to_mask
from foveated_search.search.base import confirm_prompt, point_prompt, search_config
from foveated_search.search.coords import px_to_model
from foveated_search.search.geometry import thumbnail_size
from foveated_search.search.schemas import dump_confirm, dump_point
from foveated_search.search.views import make_crop, make_thumbnail
from foveated_search.serving.images import encode_png

TYPES = ("where", "look_again", "confirm")
CODE = {"where": "w", "look_again": "a", "confirm": "c"}


def data_config() -> Config:
    return search_config(overlays=["configs/eval.yaml", "configs/train_data.yaml"])


def out_dir(cfg: Config) -> Path:
    return Path(cfg.get_key("paths.data_dir")) / cfg.get_key("train_data.out_subdir")


def images_dir(cfg: Config) -> Path:
    return Path(cfg.get_key("paths.data_dir")) / cfg.get_key("train_data.images_subdir")


def _wchoice(rng: random.Random, spec: dict):
    return rng.choices(spec["values"], weights=spec["weights"], k=1)[0]


def _box_frac_inside(b, crop) -> float:
    x, y, w, h = b
    ix = max(0.0, min(x + w, crop[2]) - max(x, crop[0]))
    iy = max(0.0, min(y + h, crop[3]) - max(y, crop[1]))
    return ix * iy / (w * h) if w * h > 0 else 0.0


def _in_xywh(x, y, b, margin=0.0) -> bool:
    return b[0] - margin <= x <= b[0] + b[2] + margin and b[1] - margin <= y <= b[1] + b[3] + margin


def _size_bucket(area: float, e: dict) -> str:
    return "small" if area < e["size_small_max"] else "medium" if area < e["size_medium_max"] else "large"


# ============================================================================ index + rules
class Rules:
    """Target / negative rules shared by planner, workers and tests."""

    def __init__(self, idx: CocoIndex, cfg: Config):
        self.idx, self.cfg = idx, cfg
        self.td = cfg["train_data"]
        self.food = cfg.get_key("eval.food_supercategory")
        self.food_cats = idx.supercat_members(self.food)
        self.cat_by_name = {c["name"]: c for c in idx.cats.values()}
        excl = set(self.td["food_negative_exclude"])
        unknown = excl - set(self.cat_by_name)
        if unknown:
            raise ValueError(f"food_negative_exclude: unknown categories {sorted(unknown)}")
        self.food_exclude = {self.cat_by_name[n]["id"] for n in excl}
        self.queries = sorted(c["name"] for c in idx.cats.values()) + [self.food]

    def target_cats(self, q: str) -> set[int]:
        return set(self.food_cats) if q == self.food else {self.cat_by_name[q]["id"]}

    def supercat_of(self, q: str) -> set[int]:
        return set(self.food_cats) if q == self.food else self.idx.supercat_members(self.cat_by_name[q]["supercategory"])

    def present(self, iid: int) -> set[int]:
        return {a["category_id"] for a in self.idx.anns_by_image.get(iid, [])}

    def targets(self, iid: int, q: str) -> list[dict]:
        tc = self.target_cats(q)
        return [a for a in self.idx.anns_by_image.get(iid, []) if a["category_id"] in tc and not a["iscrowd"]]

    def negative_ok(self, iid: int, q: str) -> bool:
        """Image-level negative (Type 1/1b absent): query's whole supercategory absent, crowd included.
        For "food": also none of the food_negative_exclude classes (unlabelled food risk)."""
        p = self.present(iid)
        if p & self.supercat_of(q):
            return False
        return not (q == self.food and p & self.food_exclude)

    def no_crop_image_ok(self, iid: int, q: str) -> bool:
        """Images allowed for Type 2 "no" crops: for "food" the same strict rule as negatives;
        otherwise any image (target absent or present elsewhere: crops are checked geometrically)."""
        if q == self.food:
            return self.negative_ok(iid, q)
        return not any(a["iscrowd"] for a in self.idx.anns_by_image.get(iid, [])
                       if a["category_id"] in self.target_cats(q))

    def positive_info(self, iid: int, q: str) -> dict | None:
        anns = [a for a in self.idx.anns_by_image.get(iid, []) if a["category_id"] in self.target_cats(q)]
        if not anns or any(a["iscrowd"] for a in anns):
            return None
        im = self.idx.images[iid]
        largest = max(anns, key=lambda a: (a["bbox"][2] * a["bbox"][3], -a["id"]))
        frac = largest["bbox"][2] * largest["bbox"][3] / (im["width"] * im["height"])
        if frac < self.td["min_box_area_frac"]:
            return None
        hard = bool((self.present(iid) & (self.food_exclude if q == self.food else self.supercat_of(q)))
                    - self.target_cats(q))
        return {"small": frac < self.td["small_box_area_frac"], "hard": hard}


def load_index(cfg: Config) -> CocoIndex:
    t0 = time.perf_counter()
    idx = CocoIndex.load(coco_dir(cfg) / cfg.get_key("train_data.source_annotations"))
    print(f"[data] train2017 annotations: {len(idx.images)} images ({time.perf_counter() - t0:.0f}s)")
    return idx


# ============================================================================ planning
def select_images(idx: CocoIndex, cfg: Config) -> list[int]:
    """Seeded per-category sample of train images (balanced coverage); never an eval image."""
    rng = random.Random(f"{cfg['seed']}:images")
    k = int(cfg.get_key("train_data.images_per_category"))
    by_cat: dict[int, set[int]] = defaultdict(set)
    for iid, anns in idx.anns_by_image.items():
        for a in anns:
            if not a["iscrowd"]:
                by_cat[a["category_id"]].add(iid)
    chosen: set[int] = set()
    for cid in sorted(by_cat):
        ids = sorted(by_cat[cid])
        chosen.update(rng.sample(ids, min(k, len(ids))))
    eval_ids = {int(x) for x in (REPO_ROOT / cfg.get_key("outputs.image_ids")).read_text().split()}
    overlap = chosen & eval_ids
    assert not overlap, f"{len(overlap)} eval images selected for training"
    return sorted(chosen)


def plan(idx: CocoIndex, cfg: Config) -> dict:
    """Deterministic list of sample jobs + train/val image split."""
    td = cfg["train_data"]
    R = Rules(idx, cfg)
    images = select_images(idx, cfg)
    rng = random.Random(f"{cfg['seed']}:plan")

    pos: dict[str, list[int]] = defaultdict(list)
    tags: dict[tuple[int, str], dict] = {}
    neg: dict[str, list[int]] = defaultdict(list)
    nocrop: dict[str, list[int]] = defaultdict(list)
    for iid in images:
        for q in R.queries:
            info = R.positive_info(iid, q)
            if info:
                pos[q].append(iid)
                tags[(iid, q)] = info
            if R.negative_ok(iid, q):
                neg[q].append(iid)
            if R.no_crop_image_ok(iid, q):
                nocrop[q].append(iid)

    uses: Counter = Counter()
    jobs: list[dict] = []

    def pick(pool: list[int], q: str, t: str, want=None) -> int | None:
        ok = lambda i: uses[(i, q, t)] < td["max_uses_per_image_query"] and (want is None or want(i))  # noqa: E731
        iid = None
        for _ in range(30):  # seeded rejection sampling (fast); exhaustive filter only as a fallback
            if not pool:
                break
            i = rng.choice(pool)
            if ok(i):
                iid = i
                break
        if iid is None:
            cand = [i for i in pool if ok(i)]
            if not cand:
                return None
            iid = rng.choice(cand)
        uses[(iid, q, t)] += 1
        return iid

    def slots(n: int, shares: list[tuple[str, float]]) -> list[str]:
        """Exact-proportion variant labels (largest remainder), shuffled."""
        exact = [(name, n * s) for name, s in shares]
        alloc = {name: int(v) for name, v in exact}
        for name, v in sorted(exact, key=lambda kv: -(kv[1] - int(kv[1])))[: n - sum(alloc.values())]:
            alloc[name] += 1
        out = [name for name, _ in shares for _ in range(alloc[name])]
        rng.shuffle(out)
        return out

    for t in TYPES:
        n = int(td["counts"][t])
        n_food = round(n * float(td["food_share"]))
        cap = max(1, math.ceil(n * float(td["max_share_per_category"])))
        cats = [q for q in R.queries if q != R.food]
        rng.shuffle(cats)
        queries = [R.food] * n_food
        counts: Counter = Counter()
        rr = 0
        while len(queries) < n:  # balanced round-robin over the 80 categories, capped
            q = cats[rr % len(cats)]
            rr += 1
            if counts[q] < cap:
                queries.append(q)
                counts[q] += 1
            if rr > 100 * n:
                raise RuntimeError("category cap too low for the requested count")
        rng.shuffle(queries)
        if t == "confirm":
            y = float(td["confirm_yes_share"])
            nk = td["confirm"]["no_kinds"]
            kinds = [(f"no_{k}", (1 - y) * w / sum(nk["weights"])) for k, w in zip(nk["values"], nk["weights"])]
            variants = slots(n, [("yes", y)] + kinds)
        else:
            ns = float(td["negative_share"][t])
            hs, ss = float(td["hard_distractor_share"]), float(td["small_target_share"])
            variants = slots(n, [("neg", ns), ("pos_hard", (1 - ns) * hs), ("pos_small", (1 - ns) * ss),
                                 ("pos", (1 - ns) * (1 - hs - ss))])
        made = 0
        for q, v in zip(queries, variants):
            iid = None
            if v == "no_near_miss" and q == R.food:
                v = "no_distractor"  # "food" "no" images may not contain food at all: no near misses
            if v == "neg":
                iid = pick(neg[q], q, t)
            elif v == "no_near_miss":  # needs the target in the image (non-food queries only)
                iid = pick(pos[q], q, t)
            elif v.startswith("no_"):
                iid = pick(nocrop[q], q, t)
            else:  # pos / pos_hard / pos_small / yes: fall back to any positive if the tagged pool is empty
                # only pos_small may use targets below the eval's minimum area
                want = {"pos_hard": lambda i, q=q: tags[(i, q)]["hard"] and not tags[(i, q)]["small"],
                        "pos_small": lambda i, q=q: tags[(i, q)]["small"]}.get(v, lambda i, q=q: not tags[(i, q)]["small"])
                iid = pick(pos[q], q, t, want) or pick(pos[q], q, t, lambda i, q=q: not tags[(i, q)]["small"])
            if iid is None:
                continue
            jobs.append({"type": t, "image_id": iid, "query": q, "variant": v})
            made += 1
        print(f"[plan] {t}: {made}/{n} jobs planned")

    for i, j in enumerate(jobs):
        j["id"] = f"{CODE[j['type']]}{i:06d}"
    used = sorted({j["image_id"] for j in jobs})
    vrng = random.Random(f"{cfg['seed']}:split")
    val = set(vrng.sample(used, round(len(used) * float(td["val_share"]))))
    for j in jobs:
        j["split"] = "val" if j["image_id"] in val else "train"
    return {"jobs": jobs, "images": used, "val_images": sorted(val), "selected_images": len(images)}


# ============================================================================ generation (workers)
_W: dict = {}


def _init_worker(cfg_dict: dict, idx: CocoIndex) -> None:
    cfg = Config(cfg_dict)
    _W.update(cfg=cfg, idx=idx, rules=Rules(idx, cfg), td=cfg["train_data"], e=cfg["eval"],
              img_dir=images_dir(cfg), out=out_dir(cfg))


class _Img:
    """One source image with lazily computed on-object points."""

    def __init__(self, iid: int):
        idx = _W["idx"]
        self.im = idx.images[iid]
        self.W, self.H = self.im["width"], self.im["height"]
        self.frame = Image.open(_W["img_dir"] / self.im["file_name"]).convert("RGB")
        self._pts: dict[int, tuple[float, float]] = {}
        self._masks: dict[int, object] = {}

    def mask(self, a: dict):
        """Bool mask (H, W); empty segmentation -> the box as a mask."""
        if a["id"] not in self._masks:
            import numpy as np

            m = seg_to_mask(a["segmentation"], self.H, self.W) if a.get("segmentation") else None
            if m is None or not m.any():
                m = np.zeros((self.H, self.W), bool)
                x, y, w, h = (int(round(v)) for v in a["bbox"])
                m[y:y + max(1, h), x:x + max(1, w)] = True
            self._masks[a["id"]] = m
        return self._masks[a["id"]]

    def point(self, a: dict) -> tuple[float, float]:
        if a["id"] not in self._pts:
            p = None
            if a.get("segmentation"):
                p = on_object_point(seg_to_mask(a["segmentation"], self.H, self.W))
            if p is None:  # empty mask: fall back to the box centre
                x, y, w, h = a["bbox"]
                p = (x + w / 2, y + h / 2)
            self._pts[a["id"]] = p
        return self._pts[a["id"]]


def _write_png(img: Image.Image, sid: str) -> str:
    rel = f"images/{sid}.png"
    (_W["out"] / rel).write_bytes(encode_png(img, int(_W["td"]["png_compress_level"])).png)
    return rel


def _sample(sid: str, job: dict, img_rel: str, sent: Image.Image, prompt_text: str, target: str, meta: dict) -> dict:
    return {"id": sid, "type": job["type"], "split": job["split"], "image": img_rel,
            "messages": [{"role": "user", "content": [{"type": "image", "image": img_rel},
                                                      {"type": "text", "text": prompt_text}]},
                         {"role": "assistant", "content": [{"type": "text", "text": target}]}],
            "meta": {"image_id": job["image_id"], "query": job["query"], "variant": job["variant"],
                     "sent_w": sent.width, "sent_h": sent.height, **meta}}


def _thumb_job(job: dict, img: _Img, rng: random.Random) -> dict | None:
    cfg, td, R = _W["cfg"], _W["td"], _W["rules"]
    q = job["query"]
    neg = job["variant"] == "neg"
    targets = [] if neg else R.targets(job["image_id"], q)
    L = _wchoice(rng, td["thumbnail_long_side"])
    choice = None
    if not neg:
        # target must be at least a few pixels in the thumbnail; grow the thumbnail if needed
        largest = max(targets, key=lambda a: (a["bbox"][2] * a["bbox"][3], -a["id"]))
        for L2 in [v for v in td["thumbnail_long_side"]["values"] if v >= L]:
            tw, th = thumbnail_size(img.W, img.H, L2)
            if max(largest["bbox"][2] * tw / img.W, largest["bbox"][3] * th / img.H) >= td["min_target_thumb_px"]:
                L = L2
                break
        else:
            return None
        choice = largest
    thumb = make_thumbnail(img.frame, L)
    tw, th = thumb.size
    checked_model: list[tuple[int, int]] = []
    checked_px: list[tuple[float, float]] = []
    avoid: list = []
    if job["type"] == "look_again":
        la = td["look_again"]
        n = _wchoice(rng, la["n_checked"])
        others = [a for a in _W["idx"].anns_by_image.get(job["image_id"], [])
                  if a["category_id"] not in R.target_cats(q) and not a["iscrowd"]]
        # "food": "no food there" must not be said about places that may hold unlabelled food, so
        # checked points avoid food-adjacent objects (bowl, cup, dining table, ...) and their boxes
        avoid = [a["bbox"] for a in others if a["category_id"] in R.food_exclude] if q == R.food else []
        if q == R.food:
            others = [a for a in others if a["category_id"] not in R.food_exclude]
        same = [a for a in others if a["category_id"] in R.supercat_of(q)]
        m = float(la["background_margin_px"])
        for _ in range(n):
            for _try in range(int(td["confirm"]["max_tries"])):
                if others and rng.random() < la["checked_on_distractor_share"]:
                    a = rng.choice(same) if same and rng.random() < 0.7 else rng.choice(others)
                    px, py = img.point(a)
                else:
                    px, py = rng.uniform(0, img.W), rng.uniform(0, img.H)
                mx, my = px_to_model(px, py, img.W, img.H)
                bx, by = mx / 1000 * img.W, my / 1000 * img.H  # where the model coords actually point
                if any(_in_xywh(bx, by, t["bbox"], m) for t in targets) or any(_in_xywh(bx, by, b, m) for b in avoid):
                    continue
                if any(abs(mx - cx) < 50 and abs(my - cy) < 50 for cx, cy in checked_model):
                    continue  # distinct checked points
                checked_model.append((mx, my))
                checked_px.append((round(bx, 2), round(by, 2)))
                break
        if not checked_model:
            return None
        if len(targets) > 1:  # prefer the instance farthest from the checked points (a "different" one)
            big_enough = (lambda a: True) if job["variant"] == "pos_small" else \
                (lambda a: a["bbox"][2] * a["bbox"][3] / (img.W * img.H) >= td["small_box_area_frac"])
            elig = [a for a in targets if max(a["bbox"][2] * tw / img.W, a["bbox"][3] * th / img.H) >= td["min_target_thumb_px"]
                    and big_enough(a)] or [choice]
            choice = max(elig, key=lambda a: (min(math.dist(img.point(a), c) for c in checked_px), -a["id"]))
    if neg:
        tgt_model, target = None, dump_point(None)
    else:
        p = img.point(choice)
        tgt_model = px_to_model(p[0], p[1], img.W, img.H)
        target = dump_point(tgt_model)
    sx, sy = tw / img.W, th / img.H
    meta = {"query_type": "supercategory" if q == R.food else "category", "polarity": "neg" if neg else "pos",
            "thumb_long_side": L, "region": [0, 0, img.W, img.H], "target_model": list(tgt_model) if tgt_model else None,
            "target_category": _W["idx"].cats[choice["category_id"]]["name"] if choice else None,
            "size_bucket": _size_bucket(choice["area"], _W["e"]) if choice else None,
            "small": bool(choice and choice["bbox"][2] * choice["bbox"][3] / (img.W * img.H) < td["small_box_area_frac"]),
            "gt_boxes_frame": [list(map(float, a["bbox"])) for a in targets],
            "gt_boxes_sent": [[round(a["bbox"][0] * sx, 2), round(a["bbox"][1] * sy, 2), round(a["bbox"][2] * sx, 2),
                               round(a["bbox"][3] * sy, 2)] for a in targets],
            "checked_model": [list(c) for c in checked_model], "checked_frame": [list(c) for c in checked_px],
            "avoid_boxes_frame": [list(map(float, b)) for b in avoid] if job["type"] == "look_again" else []}
    rel = _write_png(thumb, job["id"])
    return _sample(job["id"], job, rel, thumb, point_prompt(_W["cfg"], q, checked_model or None), target, meta)


def _crop_job(job: dict, img: _Img, rng: random.Random) -> dict | None:
    td, R = _W["td"], _W["rules"]
    c = td["confirm"]
    q = job["query"]
    targets = R.targets(job["image_id"], q)
    S = _wchoice(rng, td["crop_size"])
    yes_wanted = job["variant"] == "yes"
    label_on = rng.random() < float(c["label_share"])
    kind = None if yes_wanted else job["variant"][3:]   # planned: near_miss | distractor | background
    others = [a for a in _W["idx"].anns_by_image.get(job["image_id"], [])
              if a["category_id"] not in R.target_cats(q) and not a["iscrowd"]]
    same = [a for a in others if a["category_id"] in R.supercat_of(q)]

    def evaluate(box):
        """Per target instance: share of its mask inside the crop (vis) and share of the crop it covers.
        clear_yes: (vis, cover, instance, visible-part mask) for instances that clearly count as seen."""
        x0, y0, x1, y1 = box
        area = (x1 - x0) * (y1 - y0)
        clear_yes, clear_no, vmax = [], True, 0.0
        for t in targets:
            m = img.mask(t)
            part = m[y0:y1, x0:x1]
            inside = int(part.sum())
            vis, cover = inside / max(1, int(m.sum())), inside / area
            vmax = max(vmax, vis)
            if vis > c["no_max_visible"] or cover > c["no_max_cover"]:
                clear_no = False
            if inside and (vis >= c["yes_min_visible"] or cover >= c["yes_min_cover"]):
                ys, xs = part.nonzero()
                if max(xs.max() - xs.min(), ys.max() - ys.min()) + 1 >= c["min_target_crop_px"]:
                    clear_yes.append((vis, cover, t, part))
        return clear_yes, clear_no, vmax

    for _try in range(int(c["max_tries"])):
        if kind == "near_miss" and _try == int(c["max_tries"]) // 2:
            # no valid sliver after half the tries (box at the frame edge, or other instances inside):
            # fall back to the next "no" kind and record what the crop really is
            kind = "distractor" if others else "background"
        if yes_wanted:
            elig = [t for t in targets if max(t["bbox"][2], t["bbox"][3]) >= c["min_target_crop_px"]]
            if not elig:
                return None
            # 30% the smallest eligible instance (small targets), else the largest (random instances
            # over-sample crowds of tiny objects: 35% COCO-small in the first generation)
            t0 = min(elig, key=lambda a: (a["area"], a["id"])) if rng.random() < 0.3 else \
                max(elig, key=lambda a: (a["area"], -a["id"]))
            px, py = img.point(t0)
            span = S / 2 - c["yes_edge_margin_frac"] * S
            cx, cy = px + rng.uniform(-span, span), py + rng.uniform(-span, span)
        elif kind == "distractor" and others:
            a = rng.choice(same) if same and rng.random() < 0.7 else rng.choice(others)
            px, py = img.point(a)
            cx, cy = px + rng.uniform(-S / 6, S / 6), py + rng.uniform(-S / 6, S / 6)
        elif kind == "near_miss":
            # crop edge overlaps a sliver (2%..no_max of the box width/height) of one target box, and
            # spans the box on the other axis, so the visible share is exactly the sliver
            x, y, w, h = rng.choice(targets)["bbox"]
            side = rng.choice("lrtb")
            ov = rng.uniform(0.02, c["no_max_visible"])
            if side in "lr":
                cx = x - S / 2 + ov * w if side == "l" else x + w + S / 2 - ov * w
                slack = max(0.0, (S - h) / 2)
                cy = y + h / 2 + rng.uniform(-slack, slack)
            else:
                cy = y - S / 2 + ov * h if side == "t" else y + h + S / 2 - ov * h
                slack = max(0.0, (S - w) / 2)
                cx = x + w / 2 + rng.uniform(-slack, slack)
        else:
            cx, cy = rng.uniform(0, img.W), rng.uniform(0, img.H)
        crop, box = make_crop(img.frame, cx, cy, S)
        clear_yes, clear_no, vmax = evaluate(box)
        if yes_wanted and clear_yes:
            v, cov, t, part = max(clear_yes, key=lambda e: (e[2] is t0, e[0], e[1], e[2]["area"]))
            px_c, py_c = on_object_point(part)  # on the VISIBLE part of the object, in crop pixels
            cw, ch = box[2] - box[0], box[3] - box[1]
            pm = px_to_model(px_c, py_c, cw, ch)
            label = _W["idx"].cats[t["category_id"]]["name"] if label_on else None
            target = dump_confirm(True, pm, label)
            chosen, polarity = t, "yes"
            break
        if not yes_wanted and clear_no:
            if kind == "near_miss" and vmax == 0:
                kind = "background"  # missed the sliver: honest label of what this crop is
            target, chosen, polarity, pm = dump_confirm(False), None, "no", None
            break
    else:
        return None
    sx0, sy0 = box[0], box[1]
    meta = {"query_type": "supercategory" if q == R.food else "category", "polarity": polarity,
            "no_kind": kind, "crop_size": S, "region": list(box), "label_on": label_on,
            "target_model": list(pm) if pm else None,
            "target_category": _W["idx"].cats[chosen["category_id"]]["name"] if chosen else None,
            "size_bucket": _size_bucket(chosen["area"], _W["e"]) if chosen else None,
            "small": bool(chosen and chosen["bbox"][2] * chosen["bbox"][3] / (img.W * img.H) < td["small_box_area_frac"]),
            "gt_boxes_frame": [list(map(float, t["bbox"])) for t in targets],
            "gt_boxes_sent": [[round(t["bbox"][0] - sx0, 2), round(t["bbox"][1] - sy0, 2), round(t["bbox"][2], 2),
                               round(t["bbox"][3], 2)] for t in targets if _box_frac_inside(t["bbox"], box) > 0],
            "max_visible": round(vmax, 4), "point_crop_px": [round(px_c, 2), round(py_c, 2)] if polarity == "yes" else None}
    rel = _write_png(crop, job["id"])
    return _sample(job["id"], job, rel, crop, confirm_prompt(_W["cfg"], q, refine_point=True, label=label_on), target, meta)


def _run_image(args: tuple[int, list[dict]]) -> list[tuple[str, dict | None]]:
    iid, jobs = args
    img = _Img(iid)
    out = []
    for job in jobs:
        rng = random.Random(f"{_W['cfg']['seed']}:{job['id']}")
        fn = _crop_job if job["type"] == "confirm" else _thumb_job
        out.append((job["id"], fn(job, img, rng)))
    return out


def generate(cfg: Config, idx: CocoIndex, p: dict) -> dict:
    od = out_dir(cfg)
    (od / "images").mkdir(parents=True, exist_ok=True)
    by_img: dict[int, list[dict]] = defaultdict(list)
    for j in p["jobs"]:
        by_img[j["image_id"]].append(j)
    work = sorted(by_img.items())
    t0 = time.perf_counter()
    results: dict[str, dict | None] = {}
    with Pool(int(cfg.get_key("train_data.workers")), initializer=_init_worker, initargs=(dict(cfg), idx)) as pool:
        for i, res in enumerate(pool.imap(_run_image, work, chunksize=8), 1):
            results.update(res)
            if i % 1000 == 0:
                print(f"[gen] {i}/{len(work)} images ({time.perf_counter() - t0:.0f}s)", flush=True)
    gen_s = time.perf_counter() - t0
    samples = [results[j["id"]] for j in p["jobs"] if results[j["id"]] is not None]
    dropped = Counter(j["type"] for j in p["jobs"] if results[j["id"]] is None)
    hashes = {}
    for split in ("train", "val"):
        data = "".join(json.dumps(s, separators=(",", ":"), ensure_ascii=False) + "\n"
                       for s in samples if s["split"] == split).encode()
        (od / f"{split}.jsonl").write_bytes(data)
        hashes[f"{split}.jsonl"] = hashlib.sha256(data).hexdigest()
    print(f"[gen] {len(samples)} samples in {gen_s:.0f}s; dropped {dict(dropped)}; " +
          ", ".join(f"{k} {v[:12]}" for k, v in hashes.items()))
    return {"samples": len(samples), "dropped": dict(dropped), "generation_s": round(gen_s, 1), "sha256": hashes}


def read_samples(cfg: Config, split: str) -> list[dict]:
    with open(out_dir(cfg) / f"{split}.jsonl") as f:
        return [json.loads(line) for line in f]


# ============================================================================ CLI
def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    cfg = data_config()
    od = out_dir(cfg)
    od.mkdir(parents=True, exist_ok=True)
    idx = load_index(cfg)
    t_plan = time.perf_counter()
    p = plan(idx, cfg)
    t_plan = time.perf_counter() - t_plan
    print(f"[plan] {len(p['jobs'])} jobs on {len(p['images'])} images ({len(p['val_images'])} val), "
          f"selected {p['selected_images']}; {t_plan:.0f}s")
    meta_path = od / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    if cmd in ("fetch", "all"):
        from foveated_search.data.fetch_images import fetch_to_dir

        rows = [idx.images[i] for i in p["images"]]
        meta["fetch"] = fetch_to_dir(rows, images_dir(cfg), cfg["fetch"], workers=32)
        meta.setdefault("fetch_first", meta["fetch"])  # the run that actually downloaded (later runs skip)
        meta_path.write_text(json.dumps(meta, indent=1, default=str) + "\n")
    if cmd in ("generate", "all"):
        import subprocess

        meta.update(generate(cfg, idx, p))
        meta.update({
            "seed": cfg["seed"], "prompts_version": cfg["prompts"]["version"],
            "config": {k: cfg[k] for k in ["train_data", "prompts", "search", "server"]},
            "images_used": len(p["images"]), "val_images": len(p["val_images"]), "plan_s": round(t_plan, 1),
            "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True,
                                         text=True).stdout.strip(),
            "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        })
        meta_path.write_text(json.dumps(meta, indent=1, default=str) + "\n")
    if cmd in ("report", "all", "generate"):
        from foveated_search.data.train_data_report import samples_grid, write_stats

        tr, va = read_samples(cfg, "train"), read_samples(cfg, "val")
        write_stats(cfg, tr, va, meta)
        samples_grid(cfg, tr + va)


if __name__ == "__main__":
    main()
