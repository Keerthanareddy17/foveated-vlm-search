#!/usr/bin/env python
"""Smoke test for the demo server: one eye-search query (demo settings) on a 1280x960 mosaic if the mosaics
exist on the local disk, else on a COCO eval image; prints the answer path and whether it hit the ground truth.
Config comes from FVS_CONFIG_OVERLAYS (set by `make demo-smoke`)."""

from __future__ import annotations

import asyncio

from foveated_search.data.manifest import read_jsonl
from foveated_search.eval.frames import load_frame
from foveated_search.eval.runner import eval_config
from foveated_search.search.base import run_strategy
from foveated_search.search.geometry import in_any_xywh
from foveated_search.serving.client import VLMClient


async def main() -> None:
    cfg = eval_config()
    try:
        row = next(r for r in read_jsonl(cfg.get_key("mosaic.manifest_pattern").format(g=2)) if not r["is_negative"])
    except FileNotFoundError:
        row = next(r for r in read_jsonl(cfg.get_key("outputs.dev_manifest")) if not r["is_negative"])
    frame = load_frame(cfg, row)
    c = VLMClient(cfg)
    try:
        t = await run_strategy("eye_search", frame, row["query"], c, cfg)
    finally:
        await c.close()
    ok = bool(t.found and t.final_point and in_any_xywh(*t.final_point, row["gt_boxes"]))
    s = cfg["search"]
    print(f"model {cfg.get_key('model.id')} | frame {frame.width}x{frame.height} | query '{row['query']}' ({row['query_id']}) | "
          f"thumbnail {t.params['thumbnail']} | mask {t.params['look_again_mask']} | settings thumb {s['thumbnail_long_side']}")
    for r in t.calls:
        print(f"  {r.role:7s} fix {r.fixation} sent {r.sent_w}x{r.sent_h} vision tok {r.vision_tokens} {1000 * r.total_s:.0f} ms -> {r.raw_answer}")
    print(f"found={t.found} final_point={t.final_point} correct={ok} time-to-answer {1000 * t.totals()['time_to_answer_s']:.0f} ms")


if __name__ == "__main__":
    asyncio.run(main())
