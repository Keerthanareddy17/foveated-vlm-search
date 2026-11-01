"""Step 8b frame-size scaling: token verification, downscale budgets, throughput runs, and the per
(frame size x strategy) rows used by results/step8b_scaling.csv, the figure and results/table.md.

  python -m foveated_search.eval.scaling verify|budgets|throughput|report
"""

from __future__ import annotations

import asyncio
import csv
import json
import re
import sys

from foveated_search.config import REPO_ROOT
from foveated_search.eval.metrics import latency, summarize
from foveated_search.eval.runner import RUNS_FILE, eval_config, run_specs
from foveated_search.eval.splits import load_split

THROUGHPUT_CSV = REPO_ROOT / "results/step8b_throughput.csv"
SCALING_CSV = REPO_ROOT / "results/step8b_scaling.csv"
TPUT_N = 160          # searches per throughput measurement (10 waves at concurrency 16)
TPUT_CONC = 16
SPLIT = {1: "test_fit", 2: "test_m2", 3: "test_m3"}


def run_cfg(run: str):
    spec = run_specs()["runs"][run]
    return eval_config(spec.get("set"), spec.get("overlays")), spec


def load(run: str):
    from foveated_search.eval.table import load_run

    r = load_run(run)
    return r[0] if r else None


async def verify() -> None:
    """Full frames at every size reach the model at native resolution: processor tokens == server tokens."""
    from foveated_search.eval.frames import load_frame
    from foveated_search.search.base import point_prompt
    from foveated_search.serving.client import VLMClient

    cfg, _ = run_cfg("sc3_full")
    c = VLMClient(cfg)
    out = []
    try:
        text = point_prompt(cfg, "cat")
        base = await c.chat([], text, max_tokens=4)
        for g in (1, 2, 3):
            row = load_split(cfg, SPLIT[g])[0]
            f = load_frame(cfg, row)
            r = await c.chat([f], text, max_tokens=4)
            im = r.images[0]
            out.append({"grid": g, "sent": f"{im.width}x{im.height}", "processor_resized": f"{im.resized_w}x{im.resized_h}",
                        "processor_vision_tokens": im.vision_tokens, "server_image_tokens": r.prompt_tokens - base.prompt_tokens - 2,
                        "native": (im.resized_w, im.resized_h) == (im.width, im.height)})
    finally:
        await c.close()
    s = cfg["server"]
    res = {"image_max_pixels": s["image_max_pixels"], "max_model_len": s["max_model_len"], "frames": out,
           "all_native_and_agree": all(x["native"] and x["processor_vision_tokens"] == x["server_image_tokens"] for x in out)}
    (REPO_ROOT / "results/step8b_token_check.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))


def budgets() -> None:
    """Downscale budget per frame size = eye-96's measured mean pixels per search (written into runs.yaml)."""
    text = RUNS_FILE.read_text()
    for g in (1, 2, 3):
        tr = load(f"sc{g}_eye96")
        mean = summarize(tr)["pixels"]
        b = int(round(mean))
        text = re.sub(rf"search\.downscale\.pixel_budget: BUDGET_{g}\}}",
                      f"search.downscale.pixel_budget: {b}}}", text)
        print(f"{g}x{g}: eye-96 mean pixels {mean:.1f} -> downscale budget {b}")
    RUNS_FILE.write_text(text)


async def throughput() -> None:
    from foveated_search.eval.throughput import e2e_throughput
    from foveated_search.serving.client import VLMClient

    spec = run_specs()["scaling"]
    rows_out = []
    for size, g in spec["sizes"].items():
        for label, key in spec["strategies"].items():
            run = f"sc{g}_{key}"
            cfg, rs = run_cfg(run)
            rows = sorted(load_split(cfg, SPLIT[g]), key=lambda r: r["query_id"])[:TPUT_N]
            c = VLMClient(cfg, counter=False)
            try:
                r = await e2e_throughput(c, cfg, rs["strategy"], rows, TPUT_CONC)
            finally:
                await c.close()
            rows_out.append({"frame": size, "grid": g, "strategy": label, "run": run, **r})
            print(rows_out[-1], flush=True)
    with open(THROUGHPUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0]))
        w.writeheader()
        w.writerows(rows_out)


def scaling_rows() -> list[dict]:
    cfg = eval_config()
    spec = run_specs()["scaling"]
    h = cfg["harness"]
    fit_ids = {r["query_id"] for r in load_split(cfg, "test_fit")}
    tput = {r["run"]: r for r in csv.DictReader(open(THROUGHPUT_CSV))} if THROUGHPUT_CSV.exists() else {}
    out = []
    for size, g in spec["sizes"].items():
        for label, key in spec["strategies"].items():
            run = f"sc{g}_{key}"
            acc, lat = load(run), load(f"{run}_lat")
            if not acc:
                continue
            acc = [t for t in acc if t["meta"]["query_id"] in fit_ids]
            s = summarize(acc, int(h["bootstrap_resamples"]), float(h["ci_level"]), int(cfg["seed"]))
            la = latency(lat) if lat else {}
            tp = tput.get(run, {})
            out.append({
                "frame": size, "grid": g, "frame_pixels": acc[0]["frame_w"] * acc[0]["frame_h"], "strategy": label,
                "run": run, "n": s["n"], "found": round(s["found"], 4), "found_ci_lo": round(s["found_ci"][0], 4),
                "found_ci_hi": round(s["found_ci"][1], 4), "fp_rate": round(s["fp_rate"], 4),
                "accuracy": round(s["accuracy"], 4), "accuracy_ci_lo": round(s["accuracy_ci"][0], 4),
                "accuracy_ci_hi": round(s["accuracy_ci"][1], 4), "wrong_location": round(s["wrong_location"], 4),
                "pixels_frac": round(s["pixels_frac"], 4), "pixels": round(s["pixels"], 1),
                "vision_tokens": round(s["vision_tokens"], 1), "output_tokens": round(s["completion_tokens"], 1),
                "calls": round(s["calls"], 3), "parse_errors": s["parse_errors"],
                "lat_n": la.get("n"), "tta_p50_ms": round(1000 * la["tta_p50"], 1) if la else None,
                "tta_p95_ms": round(1000 * la["tta_p95"], 1) if la else None,
                "searches_per_s_c16": float(tp["searches_per_s"]) if tp else None,
                "accuracy_per_1k_vision_tokens": round(1000 * s["accuracy"] / s["vision_tokens"], 2),
            })
    return out


def report() -> None:
    rows = scaling_rows()
    with open(SCALING_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {SCALING_CSV.relative_to(REPO_ROOT)} ({len(rows)} rows)")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "report"
    {"verify": lambda: asyncio.run(verify()), "budgets": budgets, "throughput": lambda: asyncio.run(throughput()),
     "report": report}[cmd]()
