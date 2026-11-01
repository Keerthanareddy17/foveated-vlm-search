"""Run a strategy over a split, one trace per query, resumable.

  python -m foveated_search.eval.runner --run R4_eye_search           (named run from configs/runs.yaml)
  python -m foveated_search.eval.runner --run my_try --strategy eye_search --split dev --set search.fixation_size=160

Writes results/runs/<run>.jsonl (one trace per query) and <run>.meta.json (strategy, config snapshot,
prompts version, model, precision, git commit, split, mode, timestamps). Re-running skips queries
already in the file; it refuses to resume if the config snapshot changed.
Mode: split "latency" runs at harness.latency_concurrency (1) after warmup queries that are not
recorded; every other split is an accuracy run at harness.concurrency (latency not reported from it).
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml
from PIL import Image

from foveated_search.config import REPO_ROOT, Config
from foveated_search.data.fetch_images import eval_image_path
from foveated_search.eval.metrics import is_correct, is_found
from foveated_search.eval.splits import load_split
from foveated_search.search.base import SEARCH_OVERLAYS, get_strategy, run_strategy, search_config

RUNS_FILE = REPO_ROOT / "configs" / "runs.yaml"
META_KEYS = ["query_id", "image_id", "is_negative", "query_type", "gt_boxes", "num_instances", "size_bucket"]


def run_specs() -> dict:
    return yaml.safe_load(RUNS_FILE.read_text())


def parse_value(v: str):
    return yaml.safe_load(v)


def apply_overrides(cfg: Config, overrides: dict) -> None:
    for dotted, value in overrides.items():
        node = cfg
        *parents, leaf = dotted.split(".")
        for p in parents:
            if p not in node:
                raise KeyError(f"override {dotted!r}: no key {p!r}")
            node = node[p]
        if leaf not in node:
            raise KeyError(f"override {dotted!r}: no key {leaf!r}")
        node[leaf] = value


def eval_config(overrides: dict | None = None) -> Config:
    cfg = search_config(overlays=["configs/eval.yaml"])
    apply_overrides(cfg, overrides or {})
    return cfg


def git_commit() -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO_ROOT,
                               capture_output=True, text=True).stdout.strip()
        return sha + ("+dirty" if dirty else "")
    except Exception:
        return "unknown"


def snapshot(cfg: Config) -> dict:
    """The parts of the config that define a run's behaviour (compared on resume)."""
    return {k: cfg[k] for k in ["seed", "model", "server", "client", "search", "prompts"]}


async def run_eval(name: str, strategy: str, split: str, overrides: dict | None = None,
                   concurrency: int | None = None, limit: int | None = None) -> Path:
    cfg = eval_config(overrides)
    get_strategy(strategy)  # fail fast on unknown names
    h = cfg["harness"]
    mode = "latency" if split == "latency" else "accuracy"
    conc = int(concurrency or (h["latency_concurrency"] if mode == "latency" else h["concurrency"]))
    rows = load_split(cfg, split)[: limit or None]
    runs_dir = REPO_ROOT / h["runs_dir"]
    runs_dir.mkdir(parents=True, exist_ok=True)
    out, meta_path = runs_dir / f"{name}.jsonl", runs_dir / f"{name}.meta.json"

    snap = snapshot(cfg)
    if meta_path.exists():
        old = json.loads(meta_path.read_text())
        if old["config"] != json.loads(json.dumps(snap)) or old["strategy"] != strategy or old["split"] != split:
            raise SystemExit(f"{meta_path.name}: config/strategy/split differ from the existing run; "
                             f"use a new RUN name or delete {out.name} and {meta_path.name}")
    done = set()
    if out.exists():
        done = {json.loads(line)["meta"]["query_id"] for line in out.read_text().splitlines() if line.strip()}
    todo = [r for r in rows if r["query_id"] not in done]
    s = cfg["server"]
    meta = {
        "run": name, "strategy": strategy, "split": split, "mode": mode, "concurrency": conc,
        "n_queries": len(rows), "overrides": overrides or {}, "prompts_version": cfg["prompts"]["version"],
        "model": cfg.get_key("model.id"), "precision": s["quantization"] or s["dtype"],
        "prefix_caching": s["enable_prefix_caching"], "git_commit": git_commit(),
        "started_utc": json.loads(meta_path.read_text())["started_utc"] if meta_path.exists() else
        dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "config": snap,
    }
    meta_path.write_text(json.dumps(meta, indent=1, default=str) + "\n")
    print(f"[eval] {name}: {strategy} on {split} ({len(rows)} queries, {len(done)} done, {len(todo)} to run), "
          f"mode={mode}, concurrency={conc}")
    if not todo:
        return out

    from foveated_search.serving.client import VLMClient

    client = VLMClient(cfg)
    lock = asyncio.Lock()
    sem = asyncio.Semaphore(conc)
    n_done = 0
    t0 = time.perf_counter()

    async def one(q: dict, record: bool = True) -> None:
        nonlocal n_done
        async with sem:
            frame = Image.open(eval_image_path(cfg, q["file_name"])).convert("RGB")
            tr = await run_strategy(strategy, frame, q["query"], client, cfg)
        tr.meta = {k: q[k] for k in META_KEYS}
        d = tr.to_dict()
        d["meta"]["found_correct"] = is_found(d) if not q["is_negative"] else None
        d["meta"]["correct"] = is_correct(d)
        d["meta"]["run"] = name
        if not record:
            return
        async with lock:
            with open(out, "a") as f:
                f.write(json.dumps(d, separators=(",", ":")) + "\n")
            n_done += 1
            if n_done % 50 == 0 or n_done == len(todo):
                print(f"[eval] {name}: {n_done}/{len(todo)} ({time.perf_counter() - t0:.0f}s)", flush=True)

    try:
        if mode == "latency":
            warm = load_split(cfg, "dev")[: int(h["latency_warmup_queries"])]
            for q in warm:  # not recorded
                await one(q, record=False)
        await asyncio.gather(*(one(q) for q in todo))
    finally:
        await client.close()
    meta["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    meta["seconds_this_session"] = round(time.perf_counter() - t0, 1)
    meta_path.write_text(json.dumps(meta, indent=1, default=str) + "\n")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--strategy")
    ap.add_argument("--split")
    ap.add_argument("--set", action="append", default=[], help="dotted.key=value (repeatable or comma-separated)")
    ap.add_argument("--concurrency", type=int)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    spec = run_specs()["runs"].get(a.run, {})
    overrides = dict(spec.get("set") or {})
    for item in a.set:
        for kv in filter(None, item.split(",")):
            k, v = kv.split("=", 1)
            overrides[k.strip()] = parse_value(v)
    strategy = a.strategy or spec.get("strategy")
    split = a.split or spec.get("split")
    if not strategy or not split:
        sys.exit(f"run {a.run!r} is not in configs/runs.yaml: pass --strategy and --split")
    asyncio.run(run_eval(a.run, strategy, split, overrides, a.concurrency, a.limit))


if __name__ == "__main__":
    main()
