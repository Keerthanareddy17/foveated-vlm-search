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

SPLITS = ("dev", "test", "all", "latency")


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
    raise ValueError(f"unknown split {split!r}; known: {SPLITS}")


def split_ids(cfg: Config, split: str) -> set[str]:
    return {r["query_id"] for r in load_split(cfg, split)}
