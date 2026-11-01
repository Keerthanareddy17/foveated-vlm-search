#!/usr/bin/env python
"""Step 5 failure analysis of eye search (R4, test split, base BF16).

Every failure is categorised from the trace geometry (rules below, thresholds in configs/eval.yaml
failure_analysis), then representative trace images are saved to results/step5_failures/ for a
visual check. Writes results/step5_failures.json.

Primary category (first rule that applies):
  positives, not found
    absent_at_glance       first thumbnail point call said absent: no crop taken
    in_crop_confirm_no     some crop contained the target but the confirm said no
    off_target_small       no crop contained the target; target tiny in the thumbnail
    off_target             no crop contained the target (thumbnail point on something else)
  positives, found but point outside every gt box
    yes_point_off_in_crop  target inside the confirmed crop, refined point missed it
    yes_wrong_object       confirmed crop did not contain the target
  negatives
    false_positive         confirm said yes on a crop (target absent)
Secondary tag: gave_up_after_no = a crop was rejected and the next thumbnail call said absent.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from PIL import Image

from foveated_search.config import REPO_ROOT
from foveated_search.data.fetch_images import eval_image_path
from foveated_search.data.manifest import read_jsonl
from foveated_search.eval.metrics import is_found
from foveated_search.eval.runner import eval_config
from foveated_search.eval.splits import split_ids
from foveated_search.eval.table import load_run
from foveated_search.search.base import CallRecord, Trace
from foveated_search.search.viz import render, save_png


def box_in_crop(b, crop, frac: float) -> bool:
    x, y, w, h = b
    cx, cy = x + w / 2, y + h / 2
    if crop[0] <= cx <= crop[2] and crop[1] <= cy <= crop[3]:
        return True
    ix = max(0.0, min(x + w, crop[2]) - max(x, crop[0]))
    iy = max(0.0, min(y + h, crop[3]) - max(y, crop[1]))
    return w * h > 0 and ix * iy / (w * h) >= frac


def categorize(t: dict, fa: dict) -> tuple[str, bool] | None:
    m = t["meta"]
    calls = t["calls"]
    rejected = [i for i, c in enumerate(calls) if c["role"] == "confirm" and c["parsed"] and not c["parsed"]["y"]]
    gave_up = any(i + 1 < len(calls) and calls[i + 1]["role"] == "point" and calls[i + 1]["parsed"]
                  and calls[i + 1]["parsed"]["point"] is None for i in rejected)
    if m["is_negative"]:
        return ("false_positive", gave_up) if t["found"] else None
    if is_found(t):
        return None
    contains = [any(box_in_crop(b, c, fa["crop_contains_frac"]) for b in m["gt_boxes"]) for c in t["crops"]]
    if t["found"]:
        return ("yes_point_off_in_crop" if contains and contains[-1] else "yes_wrong_object"), gave_up
    if not t["crops"]:
        return "absent_at_glance", gave_up
    if any(contains):
        return "in_crop_confirm_no", gave_up
    tw, th = t["params"]["thumbnail"]
    side = max(max(w * tw / t["frame_w"], h * th / t["frame_h"]) for _, _, w, h in m["gt_boxes"])
    return ("off_target_small" if side < fa["small_in_thumbnail_px"] else "off_target"), gave_up


def to_trace(d: dict) -> Trace:
    t = Trace(strategy=d["strategy"], query=d["query"], frame_w=d["frame_w"], frame_h=d["frame_h"],
              params=d["params"], prompts_version=d["prompts_version"])
    t.calls = [CallRecord(**c) for c in d["calls"]]
    for k in ["found", "final_point", "fixations", "crops", "repeats", "thumbnail_sha256", "evidence", "meta"]:
        setattr(t, k, d[k])
    return t


def main() -> None:
    cfg = eval_config()
    fa = cfg["failure_analysis"]
    ids = split_ids(cfg, fa["split"])
    traces = [t for t in load_run(fa["run"])[0] if t["meta"]["query_id"] in ids]
    manifest = {r["query_id"]: r for r in read_jsonl(cfg.get_key("outputs.manifest"))}
    fails = []
    for t in traces:
        r = categorize(t, fa)
        if r:
            fails.append({"query_id": t["meta"]["query_id"], "query": t["query"], "category": r[0], "gave_up_after_no": r[1],
                          "size_bucket": t["meta"]["size_bucket"], "query_type": t["meta"]["query_type"],
                          "fixations": len(t["fixations"]), "answers": [c["raw_answer"] for c in t["calls"]]})
    cats = Counter(f["category"] for f in fails)
    n_pos = sum(not t["meta"]["is_negative"] for t in traces)
    summary = {
        "run": fa["run"], "split": fa["split"], "n_queries": len(traces), "n_pos": n_pos, "n_failures": len(fails),
        "categories": dict(cats.most_common()),
        "gave_up_after_no": sum(f["gave_up_after_no"] for f in fails),
        "positives_with_a_rejected_crop": sum(any(c["role"] == "confirm" and c["parsed"] and not c["parsed"]["y"]
                                                  for c in t["calls"]) for t in traces if not t["meta"]["is_negative"]),
        "positives_reaching_2nd_crop": sum(len(t["fixations"]) > 1 for t in traces if not t["meta"]["is_negative"]),
        "by_category_size": {c: dict(Counter(f["size_bucket"] for f in fails if f["category"] == c)) for c in cats},
        "by_category_type": {c: dict(Counter(f["query_type"] for f in fails if f["category"] == c)) for c in cats},
        "thresholds": dict(fa),
        "failures": fails,
    }
    # representative images: round-robin over categories (most frequent first), first in manifest order
    out_dir = REPO_ROOT / "results" / "step5_failures"
    out_dir.mkdir(exist_ok=True)
    for old in out_dir.glob("*.png"):
        old.unlink()
    picked, order = [], [c for c, _ in cats.most_common()]
    while len(picked) < min(int(fa["n_images"]), len(fails)):
        for c in order:
            cand = [f for f in fails if f["category"] == c and f not in picked]
            if cand and len(picked) < int(fa["n_images"]):
                picked.append(cand[0])
    by_id = {t["meta"]["query_id"]: t for t in traces}
    for f in picked:
        d = by_id[f["query_id"]]
        q = manifest[f["query_id"]]
        frame = Image.open(eval_image_path(cfg, q["file_name"])).convert("RGB")
        thumb = frame.resize(tuple(d["params"]["thumbnail"]), Image.Resampling.LANCZOS)
        lines = [f"failure category: {f['category']}" + ("  (+ gave up after a rejected crop)" if f["gave_up_after_no"] else ""),
                 "eye search calls: " + "  ->  ".join(f"{c['role']}: {c['raw_answer']}" for c in d["calls"])]
        img = render(frame, {"eye_search": to_trace(d)}, q, thumb=thumb, summary_lines=lines)
        save_png(img, out_dir / f"{f['category']}_{f['query_id']}.png")
    summary["images"] = sorted(p.name for p in out_dir.glob("*.png"))
    (REPO_ROOT / "results/step5_failures.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "failures"}, indent=1))


if __name__ == "__main__":
    main()
