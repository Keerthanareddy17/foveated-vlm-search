#!/usr/bin/env python
"""Step 9: one trace image per demo run (results/demo/<run>.json -> results/demo/traces/<run>.png).

Top: the clean photo with the hand-measured gt box (green), full frame's point and eye search's fixation path
(search/viz.py). Bottom: what eye search actually saw: the thumbnail (enlarged, nearest neighbour) and every crop,
each with the model's answer. Uses the repetition shown on the replay page (closest to the median time).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

from foveated_search.config import REPO_ROOT
from foveated_search.demo.replay import replay_config, resolve_image
from foveated_search.demo.run_demo import answer_note, notes_for, queries_spec
from foveated_search.search.base import CallRecord, Trace
from foveated_search.search.views import make_thumbnail
from foveated_search.search.viz import _font, render, save_png


def to_trace(d: dict) -> Trace:
    keys = set(Trace.__dataclass_fields__)
    t = Trace(**{k: v for k, v in d.items() if k in keys and k != "calls"})
    t.calls = [CallRecord(**c) for c in d["calls"]]
    return t


def strip(frame: Image.Image, tr: dict, thumb_long: int, height: int = 300) -> Image.Image:
    f = _font(15)
    thumb = make_thumbnail(frame, thumb_long)
    tiles = [(thumb.resize((round(thumb.width * height / thumb.height), height), Image.Resampling.NEAREST),
              f"thumbnail {thumb.width}x{thumb.height} (model view, enlarged)")]
    answers = {(c["fixation"], c["role"]): c["raw_answer"] for c in tr["calls"]}
    for k, box in enumerate(tr["crops"], 1):
        crop = frame.crop(tuple(box)).resize((height, height), Image.Resampling.LANCZOS)
        tiles.append((crop, f"crop #{k}: {answers.get((k, 'confirm'), '')}"))
    W = sum(t.width for t, _ in tiles) + 12 * (len(tiles) + 1)
    out = Image.new("RGB", (max(W, 600), height + 56), (18, 18, 22))
    d = ImageDraw.Draw(out)
    x = 12
    for im, label in tiles:
        out.paste(im, (x, 8))
        d.text((x, height + 14), label[:60], fill=(240, 240, 240), font=f)
        x += im.width + 12
    return out


def main() -> int:
    cfg = replay_config()
    runs = sorted(p for p in (REPO_ROOT / cfg.get_key("demo.out_dir")).glob("*.json"))
    out_dir = REPO_ROOT / cfg.get_key("demo.out_dir") / "traces"
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = queries_spec(cfg)
    for p in runs:
        rec = json.loads(p.read_text())
        for sm_ in rec["summary"].values():
            sm_["note"] = answer_note(sm_, rec["query"], notes_for(spec, rec["image"]))
        frame = Image.open(resolve_image(rec["image"])).convert("RGB")
        traces = {k: to_trace(v) for k, v in rec["traces"].items()}
        gt = [[x0, y0, x1 - x0, y1 - y0] for x0, y0, x1, y1 in rec.get("gt_boxes") or []]
        sm = rec["summary"]
        lines = [f"query '{rec['query']}' (expect {rec.get('expect')}), median of {rec.get('repeats', 1)} runs"] + [
            f"{s}: {sm[s]['status']} in {sm[s]['time_median_s'] if 'time_median_s' in sm[s] else sm[s]['time_to_answer_s']:.2f}s"
            f", correct {sm[s].get('n_correct')}/{sm[s].get('repeats')}: {sm[s]['evidence']}"
            + (f"  [note: {sm[s]['note']}]" if sm[s].get("note") else "") for s in sm]
        meta = {"gt_boxes": gt, "query": rec["query"], "query_id": p.stem, "is_negative": rec.get("expect") == "absent",
                "num_instances": len(gt), "size_bucket": "hand-measured box"}
        top = render(frame, traces, meta, scale=0.75, summary_lines=lines)
        bot = strip(frame, rec["traces"]["eye_search"], int(rec["settings"]["thumbnail_long_side"]))
        img = Image.new("RGB", (max(top.width, bot.width), top.height + bot.height), (18, 18, 22))
        img.paste(top, (0, 0))
        img.paste(bot, (0, top.height))
        dst = out_dir / f"{p.stem}.png"
        save_png(img, dst)
        print(f"{dst.relative_to(REPO_ROOT)} ({dst.stat().st_size / 1e3:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
