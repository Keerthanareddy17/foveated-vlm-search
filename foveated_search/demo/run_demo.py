"""Step 9: run full frame and eye search on one cleaned photo, save both traces + a summary, print a comparison.

  python -m foveated_search.demo.run_demo --image test_pictures/clean/IMG_0001.jpg --query food
  python -m foveated_search.demo.run_demo --batch          (every pair in configs/demo_queries.yaml)

Needs `make serve-demo` (fine-tuned FP8 from the private Hub repo, native-resolution frames). Config: base +
search + prompts + eval + the demo overlays (FVS_CONFIG_OVERLAYS, set by the Makefile). Timing protocol: per image,
demo.warmup_rounds unrecorded rounds of every strategy, then each strategy once, sequentially, concurrency 1.
The server has prefix caching and the mm-processor cache off, so a warmup on the same frame cannot pre-fill them.
Writes results/demo/<image stem>__<query>.json (overwritten on re-run).
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import hashlib
import json
import re
import sys
from pathlib import Path

import httpx
import yaml
from PIL import Image

from foveated_search.config import REPO_ROOT, Config
from foveated_search.eval.runner import eval_config, git_commit
from foveated_search.search.base import Trace, run_strategy
from foveated_search.serving.launch import base_url
from foveated_search.serving.client import VLMClient

SCHEMA_VERSION = 1


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-") or "x"


def run_id(image: Path, query: str) -> str:
    return f"{slug(image.stem)}__{slug(query)}"


def demo_run_config() -> Config:
    cfg = eval_config()
    for k in ["demo.strategies", "demo.warmup_rounds", "demo.expected_model", "demo.out_dir", "demo.queries_file",
              "demo.precision_label"]:
        cfg.get_key(k)
    if cfg.get_key("model.id") != cfg.get_key("demo.expected_model"):
        raise SystemExit(f"config model.id {cfg.get_key('model.id')} != demo.expected_model; "
                         "run through `make demo` (it sets the demo overlays)")
    return cfg


def served_model(cfg: Config) -> str:
    """The checkpoint the running server actually loaded (/v1/models root), so a wrong server cannot be mistaken."""
    try:
        r = httpx.get(base_url(cfg) + "/v1/models", timeout=5)
        r.raise_for_status()
    except httpx.HTTPError as e:
        raise SystemExit(f"no vLLM server at {base_url(cfg)} ({e}); start it with `make serve-demo`") from e
    return r.json()["data"][0]["root"]


def status(tr: Trace) -> str:
    return "found" if tr.found else "absent"


def evidence(tr: Trace) -> str:
    """Human-readable evidence: the confirm label (eye search) or the model's own answer (full frame)."""
    if tr.strategy == "eye_search":
        if tr.found:
            return f'confirmed on fixation {len(tr.fixations)}: "{tr.evidence}"' if tr.evidence else \
                f"confirmed on fixation {len(tr.fixations)}"
        n = len(tr.fixations)
        last = tr.calls[-1] if tr.calls else None
        checked = f"{n} crop{'s' if n != 1 else ''} checked, none confirmed; "
        if last is not None and last.role == "point":
            return (checked if n else "glance only: ") + f"no {tr.query} in the thumbnail ({last.raw_answer})"
        return checked + f"stopped after {tr.params['max_fixations']} attempts"
    return f"model answer: {tr.calls[-1].raw_answer}" if tr.calls else ""


def summarize(tr: Trace) -> dict:
    t = tr.totals()
    return {
        "strategy": tr.strategy, "status": status(tr), "final_point": tr.final_point,
        "fixations": len(tr.fixations) if tr.strategy == "eye_search" else None,
        "repeats": tr.repeats, "n_calls": t["n_calls"],
        "pixels_sent": t["pixels_sent"], "pixels_sampled_pct": round(100 * t["pixels_sampled_frac"], 2),
        "pixels_unique_pct": round(100 * t["pixels_unique_frac"], 2),
        "png_bytes": t["png_bytes"], "vision_tokens": t["vision_tokens"],
        "prompt_tokens": t["prompt_tokens"], "completion_tokens": t["completion_tokens"],
        "time_to_answer_s": round(t["time_to_answer_s"], 4), "wall_s": round(tr.wall_s, 4),
        "ttft_first_call_s": round(tr.calls[0].ttft_s, 4) if tr.calls and tr.calls[0].ttft_s else None,
        "parse_errors": t["parse_errors"], "evidence": evidence(tr),
    }


def table(rows: list[dict]) -> str:
    cols = [("strategy", "strategy", "{}"), ("status", "result", "{}"), ("fixations", "fix", "{}"),
            ("n_calls", "calls", "{}"), ("pixels_sampled_pct", "pixels %", "{:.1f}"),
            ("png_bytes", "PNG KB", "{:.0f}"), ("vision_tokens", "vision tok", "{}"),
            ("time_to_answer_s", "time (s)", "{:.3f}"), ("final_point", "final point", "{}")]
    out = []
    head = [h for _, h, _ in cols]
    body = []
    for r in rows:
        line = []
        for k, _, f in cols:
            v = r[k]
            if k == "png_bytes":
                v = v / 1e3
            if k == "final_point":
                v = f"({v[0]:.0f}, {v[1]:.0f})" if v else "-"
            line.append("-" if v is None else f.format(v))
        body.append(line)
    w = [max(len(x) for x in col) for col in zip(head, *body)]
    for line in [head, *body]:
        out.append("  ".join(x.ljust(n) for x, n in zip(line, w)))
    return "\n".join(out)


async def run_image(cfg: Config, image: Path, queries: list[str], model_root: str, quiet: bool = False) -> list[dict]:
    d = cfg["demo"]
    frame = Image.open(image).convert("RGB")
    sha = hashlib.sha256(image.read_bytes()).hexdigest()
    rel = image.resolve().relative_to(REPO_ROOT).as_posix() if image.resolve().is_relative_to(REPO_ROOT) \
        else image.as_posix()
    client = VLMClient(cfg)
    saved = []
    try:
        for _ in range(int(d["warmup_rounds"])):
            for s in d["strategies"]:
                await run_strategy(s, frame, queries[0], client, cfg)
        for q in queries:
            traces = [await run_strategy(s, frame, q, client, cfg) for s in d["strategies"]]  # sequential, conc 1
            rec = {
                "schema_version": SCHEMA_VERSION, "run_id": run_id(image, q),
                "image": rel, "image_sha256": sha, "frame_size": [frame.width, frame.height], "query": q,
                "model": model_root, "precision": d["precision_label"],
                "gpu": d.get("gpu_label"), "concurrency": 1, "warmup_rounds": int(d["warmup_rounds"]),
                "timing": "per-call total time measured by the client on the GPU host (localhost, no network); "
                          "time to answer = sum of the search's sequential call times",
                "settings": {"thumbnail_long_side": cfg.get_key("search.thumbnail_long_side"),
                             "fixation_size": cfg.get_key("search.fixation_size"),
                             "max_fixations": cfg.get_key("search.max_fixations"),
                             "look_again_mask": cfg.get_key("search.look_again_mask"),
                             "look_again_mask_rgb": cfg.get_key("search.look_again_mask_rgb"),
                             "label": cfg.get_key("search.confirm.label"),
                             "image_max_pixels": cfg.get_key("server.image_max_pixels"),
                             "png_compress_level": cfg.get_key("client.png_compress_level"),
                             "prompts_version": cfg.get_key("prompts.version")},
                "git_commit": git_commit(), "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                "summary": {tr.strategy: summarize(tr) for tr in traces},
                "traces": {tr.strategy: tr.to_dict() for tr in traces},
            }
            out = REPO_ROOT / d["out_dir"] / f"{rec['run_id']}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(rec, separators=(",", ":")) + "\n")
            saved.append(rec)
            if not quiet:
                print(f"\n{rel}  {frame.width}x{frame.height}  query '{q}'  -> {out.relative_to(REPO_ROOT)}")
                print(table(list(rec["summary"].values())))
                for s in rec["summary"].values():
                    print(f"  {s['strategy']}: {s['evidence']}")
    finally:
        await client.close()
    return saved


def batch_pairs(cfg: Config) -> dict[Path, list[str]]:
    spec = yaml.safe_load((REPO_ROOT / cfg.get_key("demo.queries_file")).read_text()) or {}
    pairs: dict[Path, list[str]] = {}
    for r in spec.get("runs") or []:
        pairs.setdefault(REPO_ROOT / r["image"], []).append(str(r["query"]))
    return pairs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--image", type=Path)
    ap.add_argument("--query", action="append", help="repeatable; all queries share one warmup")
    ap.add_argument("--batch", action="store_true", help="run every pair in demo.queries_file")
    a = ap.parse_args(argv)
    cfg = demo_run_config()
    root = served_model(cfg)
    if root != cfg.get_key("demo.expected_model"):
        raise SystemExit(f"server is serving {root}, expected {cfg.get_key('demo.expected_model')}: "
                         "make serve-stop && make serve-demo")
    if a.batch:
        pairs = batch_pairs(cfg)
        if not pairs:
            raise SystemExit(f"no runs listed in {cfg.get_key('demo.queries_file')}")
    elif a.image and a.query:
        pairs = {a.image: a.query}
    else:
        ap.error("give --image and --query, or --batch")
    for img, qs in pairs.items():
        if not img.exists():
            raise SystemExit(f"image not found: {img} (run `make intake` first)")
    print(f"server model {root} | {cfg.get_key('demo.precision_label')} | thumbnail "
          f"{cfg.get_key('search.thumbnail_long_side')} px | mask {cfg.get_key('search.look_again_mask')} | "
          f"concurrency 1, warmup rounds {cfg.get_key('demo.warmup_rounds')}")
    for img, qs in pairs.items():
        asyncio.run(run_image(cfg, img, qs, root))
    return 0


if __name__ == "__main__":
    sys.exit(main())
