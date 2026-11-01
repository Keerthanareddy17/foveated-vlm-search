"""Step 8b: high-resolution frames built from the frozen eval set (no new labels, no new downloads).

  python -m foveated_search.data.mosaic         (make mosaics: manifests + rendered frames + contact sheet)

For each test query whose image fits a 640x480 cell (never resized), a G x G mosaic of 640x480 cells
(G = 2 -> 1280x960, G = 3 -> 1920x1440): the query image in a seeded random cell, centred on neutral gray;
every other cell holds another manifest image with NO category from the query's supercategory (crowd
annotations included; for "food" also none of train_data.food_negative_exclude). Ground-truth boxes are
shifted into mosaic coordinates. Manifests are committed; rendered mosaics live on the local disk.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw

from foveated_search.config import REPO_ROOT, Config, load_config
from foveated_search.data.coco import CocoIndex, coco_dir
from foveated_search.data.fetch_images import eval_image_path
from foveated_search.data.manifest import read_jsonl


def mosaic_config() -> Config:
    return load_config(overlays=["configs/search.yaml", "configs/prompts.yaml", "configs/eval.yaml",
                                 "configs/train_data.yaml"])


def fits(r: dict, m: dict) -> bool:
    return r["width"] <= m["cell_w"] and r["height"] <= m["cell_h"]


def manifest_path(cfg: Config, grid: int) -> Path:
    return REPO_ROOT / cfg.get_key("mosaic.manifest_pattern").format(g=grid)


def build_manifest(cfg: Config, grid: int, idx: CocoIndex) -> list[dict]:
    from foveated_search.eval.splits import load_split

    m = cfg["mosaic"]
    cats = {c["name"]: c for c in idx.cats.values()}
    supercat = lambda name: {c["id"] for c in idx.cats.values() if c["supercategory"] == cats[name]["supercategory"]}  # noqa: E731
    food = cfg.get_key("eval.food_supercategory")
    food_ids = {c["id"] for c in idx.cats.values() if c["supercategory"] == food}
    excl = {cats[n]["id"] for n in cfg.get_key("train_data.food_negative_exclude")}
    present = {iid: {a["category_id"] for a in anns} for iid, anns in idx.anns_by_image.items()}
    pool = sorted((r for r in read_jsonl(cfg.get_key("outputs.manifest")) if fits(r, m)), key=lambda r: r["image_id"])
    rows = []
    for q in sorted((r for r in load_split(cfg, "test") if fits(r, m)), key=lambda r: r["query_id"]):
        blocked = (food_ids | excl) if q["query"] == food else supercat(q["query"])
        ok = [r for r in pool if r["image_id"] != q["image_id"] and not (present.get(r["image_id"], set()) & blocked)]
        rng = random.Random(f"{cfg['seed']}:mosaic:{grid}:{q['query_id']}")
        n = grid * grid
        qcell = rng.randrange(n)
        fillers = rng.sample(ok, n - 1)
        cells, fi = [], iter(fillers)
        for c in range(n):
            src = q if c == qcell else next(fi)
            cx, cy = (c % grid) * m["cell_w"], (c // grid) * m["cell_h"]
            ox, oy = (m["cell_w"] - src["width"]) // 2, (m["cell_h"] - src["height"]) // 2  # centred on gray
            cells.append({"cell": c, "image_id": src["image_id"], "file_name": src["file_name"],
                          "x": cx + ox, "y": cy + oy, "w": src["width"], "h": src["height"]})
        qc = cells[qcell]
        row = {k: q[k] for k in ["query_id", "query", "query_type", "is_negative", "gt_category_names",
                                 "num_instances", "size_bucket", "max_box_area_frac"]}
        row.update({
            "image_id": q["image_id"], "grid": grid, "width": grid * m["cell_w"], "height": grid * m["cell_h"],
            "file_name": f"{m['frames_subdir']}/{grid}x{grid}/{q['query_id']}.png", "query_cell": qcell,
            "gt_boxes": [[round(x + qc["x"], 2), round(y + qc["y"], 2), w, h] for x, y, w, h in q["gt_boxes"]],
            "cells": cells,
        })
        rows.append(row)
    return rows


def frame_path(cfg: Config, row: dict) -> Path:
    return Path(cfg.get_key("paths.data_dir")) / row["file_name"]


def render(cfg: Config, row: dict) -> Image.Image:
    m = cfg["mosaic"]
    img = Image.new("RGB", (row["width"], row["height"]), tuple(m["pad_rgb"]))
    for c in row["cells"]:
        src = Image.open(eval_image_path(cfg, c["file_name"])).convert("RGB")
        assert src.size == (c["w"], c["h"]), c
        img.paste(src, (c["x"], c["y"]))
    return img


def load_mosaic(cfg: Config, row: dict) -> Image.Image:
    p = frame_path(cfg, row)
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        render(cfg, row).save(p, compress_level=1)
    return Image.open(p).convert("RGB")


def _render_one(args) -> str:
    cfg_dict, row = args
    cfg = Config(cfg_dict)
    p = frame_path(cfg, row)
    p.parent.mkdir(parents=True, exist_ok=True)
    render(cfg, row).save(p, compress_level=1)
    return str(p)


def contact_sheet(cfg: Config, rows_by_grid: dict[int, list[dict]], per_grid: int = 3) -> None:
    from foveated_search.search.viz import _font, save_png

    W = 620
    picks = []
    for g, rows in sorted(rows_by_grid.items()):
        rng = random.Random(f"{cfg['seed']}:mosaic_sheet:{g}")
        pos = [r for r in rows if not r["is_negative"]]
        neg = [r for r in rows if r["is_negative"]]
        picks += [(g, r) for r in rng.sample(pos, per_grid - 1) + rng.sample(neg, 1)]
    f = _font(15)
    H = round(W * 0.75) + 26
    sheet = Image.new("RGB", (W * per_grid, H * len(rows_by_grid)), (18, 18, 22))
    for i, (g, r) in enumerate(picks):
        im = load_mosaic(cfg, r)
        s = W / im.width
        small = im.resize((W, round(im.height * s)), Image.Resampling.LANCZOS)
        d = ImageDraw.Draw(small)
        for x, y, w, h in r["gt_boxes"]:
            d.rectangle((x * s - 1, y * s - 1, (x + w) * s + 1, (y + h) * s + 1), outline=(40, 220, 90), width=3)
        qc = r["cells"][r["query_cell"]]
        cw, ch = cfg.get_key("mosaic.cell_w"), cfg.get_key("mosaic.cell_h")
        cx, cy = (r["query_cell"] % g) * cw, (r["query_cell"] // g) * ch
        d.rectangle((cx * s, cy * s, (cx + cw) * s - 1, (cy + ch) * s - 1), outline=(255, 176, 0), width=2)
        x0, y0 = (i % per_grid) * W, (i // per_grid) * H
        sheet.paste(small, (x0, y0 + 26))
        tag = "NEGATIVE" if r["is_negative"] else f"{r['num_instances']} inst"
        ImageDraw.Draw(sheet).text((x0 + 6, y0 + 4), f"{g}x{g} {r['width']}x{r['height']}  {r['query_id']} "
                                   f"'{r['query']}' [{tag}]  query image in cell {r['query_cell']} (amber)",
                                   fill=(255, 120, 120) if r["is_negative"] else (235, 235, 235), font=f)
    save_png(sheet, REPO_ROOT / "results/step8b_mosaic_examples.png")


def main() -> None:
    from multiprocessing import Pool

    cfg = mosaic_config()
    idx = CocoIndex.load(coco_dir(cfg) / "instances_val2017.json")
    out = {}
    for g in cfg.get_key("mosaic.grids"):
        rows = build_manifest(cfg, int(g), idx)
        data = "".join(json.dumps(r, separators=(",", ":")) + "\n" for r in rows).encode()
        manifest_path(cfg, g).write_bytes(data)
        out[g] = rows
        print(f"[mosaic] {g}x{g}: {len(rows)} queries ({sum(not r['is_negative'] for r in rows)} pos), "
              f"{rows[0]['width']}x{rows[0]['height']}; sha256 {hashlib.sha256(data).hexdigest()}")
        with Pool(8) as pool:
            paths = pool.map(_render_one, [(dict(cfg), r) for r in rows], chunksize=4)
        size = sum(Path(p).stat().st_size for p in paths)
        print(f"[mosaic] rendered {len(paths)} frames to {Path(paths[0]).parent} ({size / 1e6:.0f} MB)")
    contact_sheet(cfg, out)
    print("[mosaic] wrote results/step8b_mosaic_examples.png")


if __name__ == "__main__":
    main()
