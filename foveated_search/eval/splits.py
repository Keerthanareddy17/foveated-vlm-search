"""Evaluation splits over the frozen manifest (Step 3).

dev     = results/eval_manifest_dev.jsonl (50): all prompt / setting choices are made here only
test    = manifest minus dev (450): final numbers
all     = full manifest (500)
latency = seeded sample of test (harness.latency_subset_size), run at concurrency 1
"""

from __future__ import annotations

import random

from foveated_search.config import Config
from foveated_search.data.manifest import read_jsonl

SPLITS = ("dev", "test", "all", "latency", "test_fit", "latency_fit", "test_m2", "test_m3", "latency_m2", "latency_m3")


def load_split(cfg: Config, split: str) -> list[dict]:
    rows = read_jsonl(cfg.get_key("outputs.manifest"))
    dev_ids = {r["query_id"] for r in read_jsonl(cfg.get_key("outputs.dev_manifest"))}
    if split == "all":
        return rows
    if split == "dev":
        return [r for r in rows if r["query_id"] in dev_ids]
    test = [r for r in rows if r["query_id"] not in dev_ids]
    if split == "test":
        return test
    if split == "latency":
        n = int(cfg.get_key("harness.latency_subset_size"))
        ids = set(random.Random(int(cfg["seed"])).sample(sorted(r["query_id"] for r in test), n))
        return [r for r in test if r["query_id"] in ids]
    # Step 8b: the test queries whose image fits a mosaic cell (never resized), at 1x1 and as G x G mosaics
    if split in ("test_fit", "latency_fit"):
        m = cfg["mosaic"]
        base = test if split == "test_fit" else load_split(cfg, "latency")
        return [r for r in base if r["width"] <= m["cell_w"] and r["height"] <= m["cell_h"]]
    if split in ("test_m2", "test_m3", "latency_m2", "latency_m3"):
        g = int(split[-1])
        rows = read_jsonl(cfg.get_key("mosaic.manifest_pattern").format(g=g))
        if split.startswith("latency"):
            lat = {r["query_id"] for r in load_split(cfg, "latency")}
            rows = [r for r in rows if r["query_id"] in lat]
        return rows
    raise ValueError(f"unknown split {split!r}; known: {SPLITS}")


def split_ids(cfg: Config, split: str) -> set[str]:
    return {r["query_id"] for r in load_split(cfg, split)}
