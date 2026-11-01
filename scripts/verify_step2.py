#!/usr/bin/env python
"""Step 2 behaviour checks against the running vLLM server (run in .venv-serve).

  python scripts/verify_step2.py tokens   # processor vision tokens vs server prompt_tokens
  python scripts/verify_step2.py coords   # coordinate convention: synthetic shapes + one real COCO photo
  python scripts/verify_step2.py format   # JSON-by-prompt vs structured outputs: parse rate + latency
  python scripts/verify_step2.py resize   # resize / token cost table (default vs explicit min/max pixels)

Results go to results/step2_*.{json,csv,png}. Temperature 0 and the config seed for every call.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import io
import json
import math
import random
import statistics
import sys
from pathlib import Path

from PIL import Image, ImageDraw

from foveated_search.config import load_config
from foveated_search.data.examples import COCO_EXAMPLE_URL, fetch_coco_example
from foveated_search.search.coords import model_to_px
from foveated_search.search.parsing import POINT_RESPONSE_FORMAT, ParseError, parse_point
from foveated_search.serving.client import VLMClient
from foveated_search.serving.vision_tokens import VisionTokenCounter

CFG = load_config()
RES = Path(CFG.get_key("paths.results_dir"))
SCALE = float(CFG.get_key("client.coord_scale"))
MODEL_ID = str(CFG.get_key("model.id"))

# One real photo for the visual check (not the dataset). Two cats on a couch with two remotes.
COCO_URL = COCO_EXAMPLE_URL
COCO_QUERIES = ["the remote control on the left", "the cat on the right", "the cat on the left"]

PROMPT_JSON = (
    "Point to the {q} in the image. Answer with JSON only, no other text: "
    '{{"point": [x, y]}}, or {{"point": null}} if it is not visible.'
)
# Coordinate test: target always present, so no null option (isolates the coordinate convention).
PROMPT_POINT = 'Point to the {q} in the image. Answer with JSON only, no other text: {{"point": [x, y]}}'
PROMPT_STRUCT = "Point to the {q} in the image. If it is not visible, return null for point."
POINT_SCHEMA = POINT_RESPONSE_FORMAT

SHAPES = {"red circle": "red", "blue square": "blue", "green triangle": "green"}


def synthetic(w: int, h: int, rng: random.Random) -> tuple[Image.Image, dict[str, tuple]]:
    """White image with 3 non-overlapping shapes. Returns image and {name: (cx, cy, x0, y0, x1, y1)}."""
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    r = max(8, int(min(w, h) * 0.08))
    placed: dict[str, tuple] = {}
    for name, color in SHAPES.items():
        for _ in range(500):
            cx, cy = rng.randint(r + 2, w - r - 2), rng.randint(r + 2, h - r - 2)
            if all(math.hypot(cx - p[0], cy - p[1]) > 3 * r for p in placed.values()):
                break
        box = (cx - r, cy - r, cx + r, cy + r)
        if "circle" in name:
            d.ellipse(box, fill=color)
        elif "square" in name:
            d.rectangle(box, fill=color)
        else:
            d.polygon([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], fill=color)
            cy = cy + r / 3  # triangle centroid
        placed[name] = (cx, cy, *box)
    return img, placed


def write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        keys = list(dict.fromkeys(k for r in rows for k in r))  # union, first-seen order
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def fetch_coco() -> Image.Image:
    return fetch_coco_example(CFG, COCO_URL)


# ----------------------------------------------------------------------------- tokens
async def cmd_tokens(c: VLMClient) -> dict:
    p = c.counter.processor
    q = PROMPT_JSON.format(q="red circle")
    base = await c.chat([], q)
    rows, rng = [], random.Random(CFG["seed"])
    for w, h in [(96, 72), (224, 224), (640, 480), (333, 517), (1000, 300)]:
        img, _ = synthetic(w, h, rng)
        r = await c.chat([img], q)
        txt = p.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}],
            add_generation_prompt=True, tokenize=False)
        n_proc = p(text=[txt], images=[img], images_kwargs={"size": c.counter.size}, return_tensors="pt")["input_ids"].shape[1]
        rows.append({
            "sent": f"{w}x{h}", "resized": f"{r.images[0].resized_w}x{r.images[0].resized_h}",
            "processor_vision_tokens": r.vision_tokens,
            "server_prompt_tokens": r.prompt_tokens, "processor_prompt_tokens": n_proc,
            "server_diff_vs_text_only": r.prompt_tokens - base.prompt_tokens,
            "full_prompt_match": n_proc == r.prompt_tokens,
            "diff_equals_vision_plus_2": r.prompt_tokens - base.prompt_tokens == r.vision_tokens + 2,
        })
    ok = all(x["full_prompt_match"] and x["diff_equals_vision_plus_2"] for x in rows)
    out = {"size_setting": c.counter.size, "text_only_prompt_tokens": base.prompt_tokens, "rows": rows, "all_agree": ok}
    for x in rows:
        print(x)
    print("ALL AGREE" if ok else "MISMATCH -> stop and investigate")
    return out


# ----------------------------------------------------------------------------- coords
async def cmd_coords(c: VLMClient, tag: str = "") -> dict:
    rng = random.Random(CFG["seed"])
    sizes = [(640, 480), (480, 640), (320, 320), (1024, 512), (96, 72), (224, 224)]
    rows = []
    for w, h in sizes:
        for k in range(2):
            img, shapes = synthetic(w, h, rng)
            for name, (cx, cy, x0, y0, x1, y1) in shapes.items():
                r = await c.chat([img], PROMPT_POINT.format(q=name))
                try:
                    pt = parse_point(r.answer)
                except ParseError:
                    pt = None
                row = {"w": w, "h": h, "rep": k, "shape": name, "true_cx": round(cx, 1), "true_cy": round(cy, 1),
                       "answer": r.answer.strip().replace("\n", " "), "resized": f"{r.images[0].resized_w}x{r.images[0].resized_h}"}
                if pt:
                    rx, ry = model_to_px(*pt, w, h, SCALE)                       # hypothesis A: relative 0..1000
                    rw, rh = r.images[0].resized_w, r.images[0].resized_h
                    ax, ay = pt[0] * w / rw, pt[1] * h / rh                       # hypothesis B: abs px on resized image
                    row.update(model_x=pt[0], model_y=pt[1],
                               err_rel_px=round(math.hypot(rx - cx, ry - cy), 1),
                               err_rel_frac_diag=round(math.hypot(rx - cx, ry - cy) / math.hypot(w, h), 4),
                               err_abs_px=round(math.hypot(ax - cx, ay - cy), 1),
                               hit_rel=bool(x0 <= rx <= x1 and y0 <= ry <= y1))
                rows.append(row)
    ok_rows = [r for r in rows if "err_rel_px" in r]
    summ = {
        "n": len(rows), "parsed": len(ok_rows),
        "hit_rate_relative_1000": round(sum(r["hit_rel"] for r in ok_rows) / max(1, len(ok_rows)), 3),
        "median_err_relative_px": statistics.median(r["err_rel_px"] for r in ok_rows),
        "median_err_relative_frac_diag": statistics.median(r["err_rel_frac_diag"] for r in ok_rows),
        "median_err_absolute_px": statistics.median(r["err_abs_px"] for r in ok_rows),
        "by_size": {},
    }
    for w, h in sizes:
        rr = [r for r in ok_rows if (r["w"], r["h"]) == (w, h)]
        summ["by_size"][f"{w}x{h}"] = {
            "hit_rate": round(sum(r["hit_rel"] for r in rr) / max(1, len(rr)), 3),
            "median_err_rel_px": statistics.median(r["err_rel_px"] for r in rr) if rr else None,
            "median_err_abs_px": statistics.median(r["err_abs_px"] for r in rr) if rr else None,
        }
    write_csv(RES / f"step2_coords_synthetic{tag}.csv", rows)

    # real photo
    img = fetch_coco()
    vis = img.copy()
    d = ImageDraw.Draw(vis)
    real = []
    colors = ["#ff2d55", "#00e5ff", "#ffd60a"]
    for q, col in zip(COCO_QUERIES, colors):
        r = await c.chat([img], PROMPT_POINT.format(q=q))
        pt = parse_point(r.answer)
        px, py = model_to_px(*pt, img.width, img.height, SCALE)
        real.append({"query": q, "answer": r.answer.strip(), "model_point": pt, "px": [round(px, 1), round(py, 1)]})
        rad = 9
        d.ellipse((px - rad, py - rad, px + rad, py + rad), outline=col, width=4)
        d.line((px - 16, py, px + 16, py), fill=col, width=2)
        d.line((px, py - 16, px, py + 16), fill=col, width=2)
        d.rectangle((px + 12, py - 24, px + 14 + 7 * len(q), py - 8), fill="black")
        d.text((px + 14, py - 23), q, fill=col)
    vis.quantize(256).save(RES / f"step2_point_check{tag}.png", optimize=True)  # palette PNG: small enough to commit
    out = {"synthetic_summary": summ, "real": {"url": COCO_URL, "size": [img.width, img.height], "points": real}}
    print(json.dumps(out, indent=1))
    return out


# ----------------------------------------------------------------------------- format
async def cmd_format(c: VLMClient) -> dict:
    rng = random.Random(CFG["seed"] + 1)
    cases = []
    for i in range(8):
        w, h = rng.choice([(640, 480), (480, 640), (320, 320)])
        img, shapes = synthetic(w, h, rng)
        names = list(shapes)
        cases.append((img, shapes, names[i % 3], True))
        cases.append((img, shapes, names[(i + 1) % 3], True))
        cases.append((img, shapes, "purple star", False))  # negative: not in the image
    out = {}
    for mode in ["prompt_json", "structured_json_schema"]:
        recs = []
        for img, shapes, q, present in cases:
            if mode == "prompt_json":
                r = await c.chat([img], PROMPT_JSON.format(q=q))
            else:
                r = await c.chat([img], PROMPT_STRUCT.format(q=q), response_format=POINT_SCHEMA)
            try:
                pt, parsed = parse_point(r.answer), True
            except ParseError:
                pt, parsed = None, False
            correct = None
            if parsed:
                if present:
                    cx, cy, x0, y0, x1, y1 = shapes[q]
                    if pt is None:
                        correct = False
                    else:
                        px, py = model_to_px(*pt, img.width, img.height, SCALE)
                        correct = bool(x0 <= px <= x1 and y0 <= py <= y1)
                else:
                    correct = pt is None
            recs.append({"q": q, "present": present, "answer": r.answer, "parsed": parsed, "correct": correct,
                         "ttft_s": r.ttft_s, "total_s": r.total_s, "completion_tokens": r.completion_tokens})
        n = len(recs)
        out[mode] = {
            "n": n,
            "parse_rate": round(sum(x["parsed"] for x in recs) / n, 3),
            "correct_rate_present": round(sum(bool(x["correct"]) for x in recs if x["present"]) / sum(x["present"] for x in recs), 3),
            "correct_rate_negative": round(sum(bool(x["correct"]) for x in recs if not x["present"]) / sum(not x["present"] for x in recs), 3),
            "median_ttft_s": round(statistics.median(x["ttft_s"] for x in recs), 4),
            "median_total_s": round(statistics.median(x["total_s"] for x in recs), 4),
            "mean_completion_tokens": round(statistics.mean(x["completion_tokens"] for x in recs), 1),
            "example_answers": sorted({x["answer"].strip().replace("\n", " ") for x in recs})[:6],
            "failures": [x["answer"] for x in recs if not x["parsed"]][:5],
        }
    print(json.dumps(out, indent=1))
    return out


# ----------------------------------------------------------------------------- resize
async def cmd_resize(c: VLMClient) -> dict:
    """Measure resize + token cost under the min/max pixels the server is CURRENTLY running with
    (server.image_min_pixels / image_max_pixels; null = model defaults). Rows are merged into
    results/step2_resize.csv by setting, so run once per setting (restart the server in between).
    Server prompt_tokens confirm the processor-derived counts, i.e. the setting took effect."""
    lo, hi = CFG.get_key("server.image_min_pixels"), CFG.get_key("server.image_max_pixels")
    sname = "default" if lo is None else f"explicit min={lo} max={hi}"
    sname += f" (size={c.counter.size})"
    frame = fetch_coco()  # 640x480
    imgs = []
    for L in [64, 96, 128, 192, 224]:
        t = frame.copy(); t.thumbnail((L, L), Image.Resampling.LANCZOS)
        imgs.append((f"thumbnail long side {L}", t))
        cx, cy = frame.width // 2, frame.height // 2
        imgs.append((f"square crop {L}", frame.crop((cx - L // 2, cy - L // 2, cx - L // 2 + L, cy - L // 2 + L))))
    imgs.append(("full frame", frame))
    q = PROMPT_JSON.format(q="cat")
    base = await c.chat([], q)
    rows = []
    for label, img in imgs:
        r = await c.chat([img], q)
        im = r.images[0]
        rows.append({
            "setting": sname, "image": label, "sent": f"{im.width}x{im.height}", "sent_pixels": im.pixels,
            "resized": f"{im.resized_w}x{im.resized_h}", "resized_pixels": im.resized_w * im.resized_h,
            "vision_tokens": im.vision_tokens, "server_image_tokens": r.prompt_tokens - base.prompt_tokens - 2,
            "agree": r.prompt_tokens - base.prompt_tokens - 2 == im.vision_tokens, "png_bytes": im.png_bytes,
        })
    path = RES / "step2_resize.csv"
    old = [x for x in csv.DictReader(open(path))] if path.exists() else []
    write_csv(path, [x for x in old if x["setting"] != sname] + rows)
    for x in rows:
        print(f"{x['image']:24s} sent {x['sent']:8s} -> resized {x['resized']:8s} tokens {x['vision_tokens']:4d} "
              f"server {x['server_image_tokens']:4d} {'OK' if x['agree'] else 'MISMATCH'}  png {x['png_bytes']:6d} B")
    print(sname, "ALL AGREE" if all(x["agree"] for x in rows) else "MISMATCH")
    return {"setting": sname, "rows": rows}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["tokens", "coords", "format", "resize"])
    ap.add_argument("--tag", default="", help="suffix for the results json (e.g. _fp8_online)")
    a = ap.parse_args()
    c = VLMClient(CFG)
    try:
        if a.cmd == "tokens":
            out = await cmd_tokens(c)
        elif a.cmd == "coords":
            out = await cmd_coords(c, a.tag)
        elif a.cmd == "format":
            out = await cmd_format(c)
        else:
            out = await cmd_resize(c)
    finally:
        await c.close()
    out["server"] = {k: CFG.get_key(f"server.{k}") for k in ["quantization", "image_min_pixels", "image_max_pixels", "enable_prefix_caching"]}
    (RES / f"step2_{a.cmd}{a.tag}.json").write_text(json.dumps(out, indent=1, default=str) + "\n")
    print(f"wrote results/step2_{a.cmd}{a.tag}.json")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
