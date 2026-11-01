#!/usr/bin/env python
"""Step 8 serving micro-benchmark against the RUNNING server (whatever model/overlay it was started with).

  python scripts/step8_serving_bench.py --label bf16 [--calls thumb,crop,full] [--conc 1,4,8,16] [--e2e R1ft:full_frame,R5:eye_search]

Per-call: three realistic call types on the latency-subset images with the structured-output schemas:
  thumb = 96 px thumbnail point call, crop = 224 crop confirm call (refined point on), full = full-frame point call.
  At each concurrency: N requests (>= 4 x concurrency, distinct images), p50/p95 TTFT and total, req/s, output tok/s.
End-to-end: whole searches over the 100-query latency subset at each concurrency -> queries/s per L4.
Appends rows to results/step8_serving.csv (one header). Server-side facts (KV cache, memory) come from logs/vllm.log.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import re
import statistics
import time
from pathlib import Path

import numpy as np
from PIL import Image

from foveated_search.config import REPO_ROOT
from foveated_search.data.fetch_images import eval_image_path
from foveated_search.eval.runner import eval_config
from foveated_search.eval.splits import load_split
from foveated_search.search.base import confirm_prompt, point_prompt, run_strategy
from foveated_search.search.schemas import confirm_schema, point_schema, structured_body
from foveated_search.search.views import make_crop, make_thumbnail
from foveated_search.serving.client import VLMClient

OUT = REPO_ROOT / "results/step8_serving.csv"
FIELDS = ["label", "model", "precision", "cuda_graphs", "kind", "name", "concurrency", "n", "ttft_p50_ms", "ttft_p95_ms",
          "total_p50_ms", "total_p95_ms", "req_per_s", "out_tok_per_s", "queries_per_s", "mean_prompt_tokens",
          "mean_out_tokens", "kv_cache_tokens", "model_mem_gib", "gpu_mem_used_mib"]


def server_facts() -> dict:
    log = (REPO_ROOT / "logs/vllm.log").read_text()
    kv = re.findall(r"GPU KV cache size: ([\d,]+) tokens", log)
    mem = re.findall(r"Model loading took ([\d.]+) GiB", log)
    import subprocess

    used = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                          capture_output=True, text=True).stdout.strip().splitlines()[0]
    return {"kv_cache_tokens": int(kv[-1].replace(",", "")) if kv else None, "model_mem_gib": float(mem[-1]) if mem else None,
            "gpu_mem_used_mib": int(used)}


def pct(xs, q):
    return round(float(np.percentile(np.asarray(xs), q)) * 1000, 1)


async def call_bench(client, cfg, kind: str, rows: list[dict], conc: int, n: int) -> dict:
    F = int(cfg.get_key("search.fixation_size"))
    L = int(cfg.get_key("search.thumbnail_long_side"))
    lmax = int(cfg.get_key("search.confirm.label_max_chars"))
    reqs = []
    for i in range(n):
        q = rows[i % len(rows)]
        frame = Image.open(eval_image_path(cfg, q["file_name"])).convert("RGB")
        if kind == "thumb":
            img, text, schema = make_thumbnail(frame, L), point_prompt(cfg, q["query"]), point_schema()
        elif kind == "crop":
            img = make_crop(frame, frame.width / 2, frame.height / 2, F)[0]
            text, schema = confirm_prompt(cfg, q["query"], True, False), confirm_schema(True, False, lmax)
        else:
            img, text, schema = frame, point_prompt(cfg, q["query"]), point_schema()
        reqs.append((client.encode(img), text, schema))
    for enc, text, schema in reqs[: min(4, n)]:  # warmup
        await client.chat([enc], text, max_tokens=40, extra_body=structured_body(schema))
    sem = asyncio.Semaphore(conc)
    res = []

    async def one(r):
        async with sem:
            res.append(await client.chat([r[0]], r[1], max_tokens=40, extra_body=structured_body(r[2])))

    t0 = time.perf_counter()
    await asyncio.gather(*(one(r) for r in reqs))
    wall = time.perf_counter() - t0
    return {"n": n, "ttft_p50_ms": pct([r.ttft_s for r in res], 50), "ttft_p95_ms": pct([r.ttft_s for r in res], 95),
            "total_p50_ms": pct([r.total_s for r in res], 50), "total_p95_ms": pct([r.total_s for r in res], 95),
            "req_per_s": round(n / wall, 2), "out_tok_per_s": round(sum(r.completion_tokens for r in res) / wall, 1),
            "mean_prompt_tokens": round(statistics.mean(r.prompt_tokens for r in res), 1),
            "mean_out_tokens": round(statistics.mean(r.completion_tokens for r in res), 1)}


async def e2e_bench(client, cfg, strategy: str, rows: list[dict], conc: int) -> dict:
    frames = [(Image.open(eval_image_path(cfg, q["file_name"])).convert("RGB"), q["query"]) for q in rows]
    for f, qq in frames[:3]:  # warmup
        await run_strategy(strategy, f, qq, client, cfg)
    sem = asyncio.Semaphore(conc)
    traces = []

    async def one(f, qq):
        async with sem:
            traces.append(await run_strategy(strategy, f, qq, client, cfg))

    t0 = time.perf_counter()
    await asyncio.gather(*(one(f, qq) for f, qq in frames))
    wall = time.perf_counter() - t0
    tta = [t.totals()["time_to_answer_s"] for t in traces]
    return {"n": len(traces), "total_p50_ms": pct(tta, 50), "total_p95_ms": pct(tta, 95),
            "queries_per_s": round(len(traces) / wall, 2),
            "req_per_s": round(sum(t.totals()["n_calls"] for t in traces) / wall, 2),
            "out_tok_per_s": round(sum(t.totals()["completion_tokens"] for t in traces) / wall, 1),
            "mean_prompt_tokens": round(statistics.mean(t.totals()["prompt_tokens"] for t in traces), 1),
            "mean_out_tokens": round(statistics.mean(t.totals()["completion_tokens"] for t in traces), 1)}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="configuration label, e.g. bf16 / fp8 / fp8_eager")
    ap.add_argument("--calls", default="thumb,crop,full")
    ap.add_argument("--conc", default="1,4,8,16")
    ap.add_argument("--e2e", default="", help="name:strategy pairs, e.g. R1ft:full_frame,R5:eye_search")
    ap.add_argument("--e2e-conc", default="1,4,8,16")
    a = ap.parse_args()
    cfg = eval_config()
    rows = load_split(cfg, "latency")
    s = cfg["server"]
    base = {"label": a.label, "model": cfg.get_key("model.id").split("/")[-1], "precision": s["quantization"] or
            ("fp8_dynamic (checkpoint)" if "fp8" in str(cfg.get_key("model.id")) else s["dtype"]),
            "cuda_graphs": not s["enforce_eager"], **server_facts()}
    # no processor token counting in the timed path: it runs synchronously in the event loop and would throttle
    # throughput under concurrency (vision-token counts come from the accuracy runs)
    client = VLMClient(cfg, counter=False)
    out = []
    try:
        for kind in filter(None, a.calls.split(",")):
            for c in map(int, a.conc.split(",")):
                r = await call_bench(client, cfg, kind, rows, c, max(32, 4 * c))
                out.append({**base, "kind": "call", "name": kind, "concurrency": c, **r})
                print(out[-1], flush=True)
        for pair in filter(None, a.e2e.split(",")):
            name, strategy = pair.split(":")
            for c in map(int, a.e2e_conc.split(",")):
                r = await e2e_bench(client, cfg, strategy, rows, c)
                out.append({**base, "kind": "e2e", "name": name, "concurrency": c, **r})
                print(out[-1], flush=True)
    finally:
        await client.close()
    new = not OUT.exists()
    with open(OUT, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerows(out)
    print(f"appended {len(out)} rows to {OUT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    asyncio.run(main())
