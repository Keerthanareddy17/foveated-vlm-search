"""End-to-end search throughput against the running server: completed searches per second at a given
concurrency. Frames are pre-loaded; image encoding runs off the event loop (VLMClient.aencode, Step 8);
the client does no token counting in the timed path."""

from __future__ import annotations

import asyncio
import time

import numpy as np

from foveated_search.config import Config
from foveated_search.eval.frames import load_frame
from foveated_search.search.base import run_strategy


async def e2e_throughput(client, cfg: Config, strategy: str, rows: list[dict], conc: int, warmup: int = 3) -> dict:
    frames = [(load_frame(cfg, q), q["query"]) for q in rows]
    for f, qq in frames[:warmup]:
        await run_strategy(strategy, f, qq, client, cfg)
    sem = asyncio.Semaphore(conc)
    traces = []

    async def one(f, qq):
        async with sem:
            traces.append(await run_strategy(strategy, f, qq, client, cfg))

    t0 = time.perf_counter()
    await asyncio.gather(*(one(f, qq) for f, qq in frames))
    wall = time.perf_counter() - t0
    tta = np.asarray([t.totals()["time_to_answer_s"] for t in traces])
    return {"n": len(traces), "concurrency": conc, "wall_s": round(wall, 2),
            "searches_per_s": round(len(traces) / wall, 2),
            "calls_per_s": round(sum(t.totals()["n_calls"] for t in traces) / wall, 2),
            "tta_p50_ms": round(1000 * float(np.percentile(tta, 50)), 1),
            "tta_p95_ms": round(1000 * float(np.percentile(tta, 95)), 1)}
