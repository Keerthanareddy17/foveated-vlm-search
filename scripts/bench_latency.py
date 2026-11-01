#!/usr/bin/env python
"""Per-call latency at concurrency 1: thumbnail (96 px) vs fixation crop (224 px) vs full frame.

  python scripts/bench_latency.py [--out results/step2_latency.csv]

Same fixed photo, same structured point prompt for every image type, temperature 0.
Latency measured on the same VM as the server (effectively server-side; no WAN).
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import statistics
from pathlib import Path

import numpy as np
from PIL import Image

from foveated_search.config import load_config
from foveated_search.data.examples import fetch_coco_example
from foveated_search.search.parsing import POINT_RESPONSE_FORMAT
from foveated_search.serving.client import VLMClient


def pct(xs: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(xs), q))


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/step2_latency.csv")
    a = ap.parse_args()
    cfg = load_config()
    b = cfg["bench"]
    frame = fetch_coco_example(cfg)
    L, F = int(cfg.get_key("search.thumbnail_long_side")), int(cfg.get_key("search.fixation_size"))
    thumb = frame.copy()
    thumb.thumbnail((L, L), Image.Resampling.LANCZOS)
    cx, cy = frame.width // 2, frame.height // 2
    crop = frame.crop((cx - F // 2, cy - F // 2, cx - F // 2 + F, cy - F // 2 + F))
    images = [(f"thumbnail {L}px", thumb), (f"fixation crop {F}px", crop), ("full frame", frame)]
    prompt = f"Point to the {b['query']} in the image. If it is not visible, return null for point."

    c = VLMClient(cfg)
    rows = []
    try:
        for label, img in images:
            enc = c.encode(img)
            for _ in range(int(b["warmup_calls"])):
                await c.chat([enc], prompt, max_tokens=int(b["max_tokens"]), response_format=POINT_RESPONSE_FORMAT)
            rs = [await c.chat([enc], prompt, max_tokens=int(b["max_tokens"]), response_format=POINT_RESPONSE_FORMAT)
                  for _ in range(int(b["calls"]))]
            ttft, tot = [r.ttft_s for r in rs], [r.total_s for r in rs]
            im = rs[0].images[0]
            rows.append({
                "image": label, "sent": f"{im.width}x{im.height}", "model_input": f"{im.resized_w}x{im.resized_h}",
                "pixels": im.pixels, "png_bytes": im.png_bytes, "vision_tokens": rs[0].vision_tokens,
                "prompt_tokens": rs[0].prompt_tokens, "output_tokens_mean": round(statistics.mean(r.completion_tokens for r in rs), 1),
                "n": len(rs),
                "ttft_p50_ms": round(pct(ttft, 50) * 1000, 1), "ttft_p95_ms": round(pct(ttft, 95) * 1000, 1),
                "total_p50_ms": round(pct(tot, 50) * 1000, 1), "total_p95_ms": round(pct(tot, 95) * 1000, 1),
                "answer": rs[0].answer.strip(),
            })
    finally:
        await c.close()

    s = cfg["server"]
    meta = {"quantization": s["quantization"] or "none (bf16)", "enforce_eager": s["enforce_eager"],
            "prefix_caching": s["enable_prefix_caching"], "min_pixels": s["image_min_pixels"],
            "max_pixels": s["image_max_pixels"], "concurrency": 1, "model": cfg.get_key("model.id")}
    for r in rows:
        r["server"] = json.dumps(meta, sort_keys=True)
    out = Path(a.out)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    cols = ["image", "sent", "model_input", "vision_tokens", "png_bytes", "output_tokens_mean",
            "ttft_p50_ms", "ttft_p95_ms", "total_p50_ms", "total_p95_ms"]
    print(f"server: {meta}")
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for r in rows:
        print("| " + " | ".join(str(r[k]) for k in cols) + " |")
    print(f"wrote {out}")


if __name__ == "__main__":
    asyncio.run(main())
