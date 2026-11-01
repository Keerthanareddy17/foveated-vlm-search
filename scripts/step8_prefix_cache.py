#!/usr/bin/env python
"""Step 8: prefix-cache accounting for R7 (FP8 + prefix caching + mm-processor cache).

Per-call cached prompt tokens come from vLLM (--enable-prompt-tokens-details -> usage.prompt_tokens_details);
server totals come from /metrics snapshots (logs/r7_metrics.txt, copied to results/step8_r7_metrics.txt).
Writes results/step8_prefix_cache.json and prints a table by call type / fixation.
"""

from __future__ import annotations

import json
import re
import shutil
from collections import defaultdict

import numpy as np

from foveated_search.config import REPO_ROOT
from foveated_search.eval.metrics import latency
from foveated_search.eval.table import load_run


def by_call(traces: list[dict]) -> dict:
    g = defaultdict(lambda: {"calls": 0, "prompt_tokens": 0, "cached_tokens": 0, "calls_with_hit": 0, "total_s": []})
    for t in traces:
        for c in t["calls"]:
            key = "thumbnail point, fixation 1" if c["role"] == "point" and c["fixation"] == 1 else \
                  "thumbnail point, fixations 2-3" if c["role"] == "point" else "crop confirm"
            x = g[key]
            x["calls"] += 1
            x["prompt_tokens"] += c["prompt_tokens"] or 0
            x["cached_tokens"] += c.get("cached_tokens") or 0
            x["calls_with_hit"] += bool(c.get("cached_tokens"))
            x["total_s"].append(c["total_s"])
    out = {}
    for k, x in g.items():
        out[k] = {"calls": x["calls"], "prompt_tokens": x["prompt_tokens"], "cached_tokens": x["cached_tokens"],
                  "token_hit_rate": x["cached_tokens"] / x["prompt_tokens"] if x["prompt_tokens"] else None,
                  "calls_with_any_hit": x["calls_with_hit"], "mean_call_ms": round(1000 * float(np.mean(x["total_s"])), 1)}
    return out


def metric_deltas(path) -> dict:
    snaps = defaultdict(dict)
    for line in open(path):
        tag, name, value = re.match(r"(\S+) (vllm:[a-z_]+)\{.*\} ([\d.]+)", line).groups()
        snaps[tag][name] = float(value)

    def delta(a, b):
        q = snaps[b]["vllm:prefix_cache_queries_total"] - snaps[a]["vllm:prefix_cache_queries_total"]
        h = snaps[b]["vllm:prefix_cache_hits_total"] - snaps[a]["vllm:prefix_cache_hits_total"]
        mq = snaps[b]["vllm:mm_cache_queries_total"] - snaps[a]["vllm:mm_cache_queries_total"]
        mh = snaps[b]["vllm:mm_cache_hits_total"] - snaps[a]["vllm:mm_cache_hits_total"]
        return {"prefix_tokens_queried": q, "prefix_tokens_hit": h, "prefix_hit_rate": h / q if q else None,
                "mm_items_queried": mq, "mm_items_hit": mh, "mm_hit_rate": mh / mq if mq else None}

    return {"test_run_cold_cache": delta("before", "after_test"),
            "latency_run_after_test_run_CONTAMINATED": delta("after_test", "after_lat"),
            "latency_run_cold_cache": delta("lat_before", "lat_after_cold")}


def main() -> None:
    shutil.copyfile(REPO_ROOT / "logs/r7_metrics.txt", REPO_ROOT / "results/step8_r7_metrics.txt")
    r7, r7l, r6l = load_run("R7_eye_search")[0], load_run("R7_eye_search_lat")[0], load_run("R6_eye_search_lat")[0]
    out = {"server_metrics": metric_deltas(REPO_ROOT / "results/step8_r7_metrics.txt"),
           "per_call_R7_test": by_call(r7), "per_call_R7_latency_subset": by_call(r7l),
           "latency_R6_vs_R7_conc1": {"R6": latency(r6l), "R7": latency(r7l)}}
    (REPO_ROOT / "results/step8_prefix_cache.json").write_text(json.dumps(out, indent=1, default=str) + "\n")
    for k, v in out["server_metrics"].items():
        print(f"{k:42s} prefix hit {100 * v['prefix_hit_rate']:.1f}% ({v['prefix_tokens_hit']:.0f}/{v['prefix_tokens_queried']:.0f} tokens); "
              f"image cache {100 * v['mm_hit_rate']:.1f}% ({v['mm_items_hit']:.0f}/{v['mm_items_queried']:.0f})")
    print("\nR7 test, per call type:")
    for k, v in out["per_call_R7_test"].items():
        print(f"  {k:32s} calls {v['calls']:4d}  prompt tok {v['prompt_tokens']:6d}  cached {v['cached_tokens']:5d} "
              f"({100 * (v['token_hit_rate'] or 0):.1f}%), calls with a hit {v['calls_with_any_hit']}, mean {v['mean_call_ms']} ms")
    a, b = out["latency_R6_vs_R7_conc1"]["R6"], out["latency_R6_vs_R7_conc1"]["R7"]
    print(f"\nlatency subset (conc 1) time-to-answer p50/p95: R6 {1000 * a['tta_p50']:.0f}/{1000 * a['tta_p95']:.0f} ms, "
          f"R7 {1000 * b['tta_p50']:.0f}/{1000 * b['tta_p95']:.0f} ms")


if __name__ == "__main__":
    main()
