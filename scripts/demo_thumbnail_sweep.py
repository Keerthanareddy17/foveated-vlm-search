#!/usr/bin/env python
"""Step 9: eye-search thumbnail sweep on the demo photos (an illustration on a few photos, not a benchmark).

For every (image, query) in configs/demo_queries.yaml and every demo.sweep.thumbnail_sizes value: eye search at the
demo settings with only search.thumbnail_long_side changed, demo.sweep.warmup_rounds unrecorded rounds per image and
size, then demo.sweep.repeats measured runs, sequentially, concurrency 1. Full-frame rows are copied from the saved
demo runs (results/demo/<run>.json), since full frame never sees the thumbnail.
Writes demo.sweep.out_csv (one row per image, query, strategy/size) and demo.sweep.out_runs (every measured run).
"""

from __future__ import annotations

import asyncio
import csv
import json
import statistics
import sys

from PIL import Image

from foveated_search.config import REPO_ROOT
from foveated_search.demo.run_demo import (answer_note, batch_pairs, demo_run_config, median_index, run_id, score,
                                           served_model, summarize)
from foveated_search.search.base import run_strategy
from foveated_search.search.geometry import thumbnail_size
from foveated_search.serving.client import VLMClient

COLS = ["image", "query", "expect", "strategy", "thumbnail_long_side", "thumbnail", "target_box", "target_in_thumb_px",
        "status", "correct", "n_correct", "repeats", "answers_identical", "looks", "calls", "vision_tokens",
        "png_bytes", "pixels_sampled_pct", "time_median_s", "time_min_s", "time_max_s", "final_point", "evidence", "note"]


def in_thumb(box, frame_wh, thumb_wh) -> str:
    if not box:
        return ""
    sx, sy = thumb_wh[0] / frame_wh[0], thumb_wh[1] / frame_wh[1]
    return f"{round((box[2] - box[0]) * sx)}x{round((box[3] - box[1]) * sy)}"


def row(img, q, sm_list, strategy, size, thumb, frame_wh, notes) -> dict:
    times = [s["time_to_answer_s"] for s in sm_list]
    rep = sm_list[median_index(times)]
    box = (q.get("gt_boxes") or [None])[0]
    return {"image": img.name, "query": q["query"], "expect": q.get("expect"), "strategy": strategy,
            "thumbnail_long_side": size, "thumbnail": f"{thumb[0]}x{thumb[1]}" if thumb else "",
            "target_box": json.dumps(box) if box else "", "target_in_thumb_px": in_thumb(box, frame_wh, thumb) if thumb else "",
            "status": rep["status"], "correct": rep["correct"],
            "n_correct": sum(bool(s["correct"]) for s in sm_list), "repeats": len(sm_list),
            "answers_identical": all(s["answers"] == sm_list[0]["answers"] for s in sm_list),
            "looks": rep["fixations"] if rep["fixations"] is not None else "", "calls": rep["n_calls"],
            "vision_tokens": rep["vision_tokens"], "png_bytes": rep["png_bytes"],
            "pixels_sampled_pct": round(rep["pixels_sampled_pct"], 2),
            "time_median_s": round(statistics.median(times), 4), "time_min_s": min(times), "time_max_s": max(times),
            "final_point": json.dumps([round(v) for v in rep["final_point"]]) if rep["final_point"] else "",
            "evidence": rep["evidence"], "note": answer_note(rep, q["query"], notes) or ""}


async def sweep_image(cfg, img, queries, sizes, n_rep, n_warm, runs_out) -> list[dict]:
    frame = Image.open(img).convert("RGB")
    rows = []
    client = VLMClient(cfg)
    try:
        for size in sizes:
            cfg["search"]["thumbnail_long_side"] = int(size)
            thumb = thumbnail_size(frame.width, frame.height, int(size))
            for _ in range(n_warm):
                await run_strategy("eye_search", frame, queries[0]["query"], client, cfg)
            for q in queries:
                sms = []
                for k in range(n_rep):
                    tr = await run_strategy("eye_search", frame, q["query"], client, cfg)
                    sm = summarize(tr)
                    sm["correct"] = score(sm, q.get("expect"), q.get("gt_boxes"))
                    sm["answers"] = [c.raw_answer for c in tr.calls]
                    sms.append(sm)
                    runs_out.write(json.dumps({"image": img.name, "query": q["query"], "thumbnail_long_side": size,
                                               "rep": k, "summary": sm, "trace": tr.to_dict()}, separators=(",", ":")) + "\n")
                r = row(img, q, sms, "eye_search", size, thumb, frame.size, q.get("notes"))
                rows.append(r)
                print(f"  {img.name:14s} {q['query']:6s} thumb {size} ({r['thumbnail']}, target {r['target_in_thumb_px'] or '-'}): "
                      f"{r['status']:6s} correct {r['n_correct']}/{n_rep}  looks {r['looks']}  tok {r['vision_tokens']}  "
                      f"{r['png_bytes'] / 1e3:.0f} KB  median {r['time_median_s']:.3f}s"
                      + ("" if r["answers_identical"] else "  (answers differ)") + (f"  NOTE {r['note']}" if r["note"] else ""))
    finally:
        await client.close()
    return rows


def full_frame_rows(img, queries, frame_wh) -> list[dict]:
    rows = []
    for q in queries:
        rec = json.loads((REPO_ROOT / "results" / "demo" / f"{run_id(img, q['query'])}.json").read_text())
        sms = rec["per_repeat"]["full_frame"]
        rows.append(row(img, q, sms, "full_frame", "", None, frame_wh, q.get("notes")))
    return rows


def main() -> int:
    cfg = demo_run_config()
    sw = cfg.get_key("demo.sweep")
    if served_model(cfg) != cfg.get_key("demo.expected_model"):
        raise SystemExit("wrong model served: make serve-stop && make serve-demo")
    pairs = batch_pairs(cfg)
    out_csv, out_runs = REPO_ROOT / sw["out_csv"], REPO_ROOT / sw["out_runs"]
    print(f"eye-search thumbnail sweep {sw['thumbnail_sizes']} | {sw['repeats']} runs after {sw['warmup_rounds']} warmup "
          f"per image and size | concurrency 1 | mask {cfg.get_key('search.look_again_mask')}")
    rows = []
    with open(out_runs, "w") as runs_out:
        for img, qs in pairs.items():
            with Image.open(img) as im:
                rows += full_frame_rows(img, qs, im.size)
            rows += asyncio.run(sweep_image(cfg, img, qs, sw["thumbnail_sizes"], int(sw["repeats"]),
                                            int(sw["warmup_rounds"]), runs_out))
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out_csv.relative_to(REPO_ROOT)} ({len(rows)} rows) and {out_runs.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
