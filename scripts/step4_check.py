#!/usr/bin/env python
"""Step 4 live check: all four strategies on 5 dev-manifest queries, against the running server.

  python scripts/step4_check.py          (make step4-check)

Writes results/step4_traces.jsonl, results/step4_trace_<query_id>.png, results/step4_check.json.
Also measures (a) the output-token cost of the confirm schema's optional fields and of JSON
whitespace... (b) processor vision tokens vs server prompt-token difference.
"""

from __future__ import annotations

import asyncio
import json
import statistics
from collections import defaultdict

from PIL import Image

from foveated_search.config import REPO_ROOT
from foveated_search.data.fetch_images import eval_image_path
from foveated_search.data.manifest import read_jsonl
from foveated_search.search.base import Trace, prompt, run_strategy, search_config
from foveated_search.search.geometry import in_any_xywh, thumbnail_size
from foveated_search.search.schemas import confirm_schema, point_schema, structured_body
from foveated_search.search.viz import render, save_png
from foveated_search.serving.client import VLMClient

STRATEGIES = ["full_frame", "downscale", "sliding_window", "eye_search"]
N = 5


def pick_queries(dev: list[dict]) -> list[dict]:
    """Deterministic: first negative, first two food positives, first two category positives (dev order)."""
    neg = [r for r in dev if r["is_negative"]][:1]
    food = [r for r in dev if not r["is_negative"] and r["query_type"] == "supercategory"][:2]
    cat = [r for r in dev if not r["is_negative"] and r["query_type"] == "category"][: N - len(neg) - len(food)]
    ids = {r["query_id"] for r in neg + food + cat}
    return [r for r in dev if r["query_id"] in ids]


def correct(t: Trace, q: dict) -> bool:
    if q["is_negative"]:
        return not t.found
    return bool(t.found and t.final_point and in_any_xywh(*t.final_point, q["gt_boxes"]))


async def field_costs(client: VLMClient, cfg, traces: list[Trace], frames: dict) -> dict:
    """Re-ask every confirm crop with each confirm-schema variant: same crops, only the answer
    format differs, so the token difference is the cost of each optional field."""
    out: dict = {"confirm": defaultdict(list)}
    lmax = int(cfg.get_key("search.confirm.label_max_chars"))
    variants = {"y only": (False, False), "y+p": (True, False), "y+l": (False, True), "y+p+l": (True, True)}
    for t in traces:
        if t.strategy != "eye_search":
            continue
        frame = frames[t.meta["query_id"]]
        for c in t.calls:
            if c.role == "confirm":
                crop = client.encode(frame.crop(tuple(c.region)))
                for name, (rp, lb) in variants.items():
                    fields = (prompt(cfg, "confirm_field_point") if rp else "") + (prompt(cfg, "confirm_field_label") if lb else "")
                    r = await client.chat([crop], prompt(cfg, "confirm", query=t.query, fields=fields), max_tokens=40,
                                          extra_body=structured_body(confirm_schema(rp, lb, lmax)))
                    yes = '"y":true' in r.answer.replace(" ", "")
                    out["confirm"][(name, "yes" if yes else "no")].append(r.completion_tokens)
    res = {"confirm_completion_tokens": {f"{k[0]} ({k[1]})": {"n": len(v), "mean": round(statistics.mean(v), 2)}
                                         for k, v in sorted(out["confirm"].items())}}
    return res


async def verify_tokens(client: VLMClient, t: Trace, frame: Image.Image) -> list[dict]:
    """For each call of one eye-search trace: server prompt_tokens(with image) - prompt_tokens(text
    only) must equal processor vision tokens + 2 (<vision_start>, <vision_end>), as in Step 2."""
    rows = []
    for c in t.calls:
        r = await client.chat([], c.prompt, max_tokens=8)
        diff = c.prompt_tokens - r.prompt_tokens
        rows.append({"role": c.role, "fixation": c.fixation, "sent": f"{c.sent_w}x{c.sent_h}",
                     "resized": f"{c.resized_w}x{c.resized_h}", "processor_vision_tokens": c.vision_tokens,
                     "server_diff": diff, "agree": diff == c.vision_tokens + 2})
    return rows


async def main() -> None:
    cfg = search_config(overlays=["configs/eval.yaml"])  # eval.yaml: where the eval images live
    dev = read_jsonl("results/eval_manifest_dev.jsonl")
    queries = pick_queries(dev)
    client = VLMClient(cfg)
    traces: list[Trace] = []
    frames: dict[str, Image.Image] = {}
    try:
        for q in queries:
            frame = Image.open(eval_image_path(cfg, q["file_name"])).convert("RGB")
            frames[q["query_id"]] = frame
            for name in STRATEGIES:  # sequential, concurrency 1: clean per-call latency
                t = await run_strategy(name, frame, q["query"], client, cfg)
                t.meta = {k: q[k] for k in ["query_id", "image_id", "is_negative", "query_type", "gt_boxes",
                                            "num_instances", "size_bucket"]}
                t.meta["correct"] = correct(t, q)
                traces.append(t)
        eye = [t for t in traces if t.strategy == "eye_search"]
        multi = [t for t in eye if sum(c.role == "point" for c in t.calls) > 1]
        tok_trace = max(eye, key=lambda t: len(t.calls))
        tokens = await verify_tokens(client, tok_trace, frames[tok_trace.meta["query_id"]])
        costs = await field_costs(client, cfg, traces, frames)
        # Byte-identity sweep: eye search on every dev query. run_strategy asserts the thumbnail
        # hash is identical across fixations; any violation raises here. Only counts are kept.
        sweep = {"queries": 0, "multi_fixation": 0, "repeats": 0, "max_point_calls": 0}
        for q in dev:
            f = Image.open(eval_image_path(cfg, q["file_name"])).convert("RGB")
            t = await run_strategy("eye_search", f, q["query"], client, cfg)
            n_pt = sum(c.role == "point" for c in t.calls)
            assert len({c.image_sha256 for c in t.calls if c.role == "point"}) == 1
            sweep["queries"] += 1
            sweep["multi_fixation"] += n_pt > 1
            sweep["repeats"] += t.repeats
            sweep["max_point_calls"] = max(sweep["max_point_calls"], n_pt)
    finally:
        await client.close()

    (REPO_ROOT / "results/step4_traces.jsonl").write_text("".join(t.to_json() + "\n" for t in traces))

    # ---------------------------------------------------------------- summary
    hdr = ["query_id", "query", "neg", "strategy", "calls", "decision", "correct", "pixels %", "uniq px %",
           "vis tok", "out tok", "time-to-answer ms"]
    lines = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for t in traces:
        tot = t.totals()
        lines.append("| " + " | ".join(str(v) for v in [
            t.meta["query_id"], t.query, "Y" if t.meta["is_negative"] else "", t.strategy, tot["n_calls"],
            "found" if t.found else "not found", "yes" if t.meta["correct"] else "NO",
            f"{100 * tot['pixels_sampled_frac']:.1f}", f"{100 * tot['pixels_unique_frac']:.1f}", tot["vision_tokens"],
            tot["completion_tokens"], f"{1000 * tot['time_to_answer_s']:.0f}"]) + " |")
    per_role: dict[str, list[int]] = defaultdict(list)
    for t in traces:
        for c in t.calls:
            per_role[f"{t.strategy}:{c.role}"].append(c.completion_tokens)
    role_lines = ["| strategy:call | n | mean completion tokens | mean total ms |", "|---|---|---|---|"]
    for k in sorted(per_role):
        ms = [c.total_s for t in traces for c in t.calls if f"{t.strategy}:{c.role}" == k]
        role_lines.append(f"| {k} | {len(per_role[k])} | {statistics.mean(per_role[k]):.2f} | {1000 * statistics.mean(ms):.0f} |")
    print("\n".join(lines))
    print()
    print("\n".join(role_lines))
    print("\nconfirm-field token cost (same crops re-asked):", json.dumps(costs, indent=1))
    print(f"\nthumbnail byte-identity: held on all {len(multi)} multi-fixation eye searches of the 5 queries; "
          f"dev sweep ({sweep['queries']} queries): held on all {sweep['multi_fixation']} multi-fixation searches, "
          f"{sweep['repeats']} repeats, max {sweep['max_point_calls']} point calls")
    print("vision tokens vs server (trace", tok_trace.meta["query_id"], "):")
    for r in tokens:
        print("  ", r)
    print("ALL AGREE" if all(r["agree"] for r in tokens) else "MISMATCH -> stop and investigate")

    # ---------------------------------------------------------------- visualizations
    for q in queries:
        qid = q["query_id"]
        tq = {t.strategy: t for t in traces if t.meta["query_id"] == qid}
        frame = frames[qid]
        thumb = frame.resize(tuple(tq["eye_search"].params["thumbnail"]), Image.Resampling.LANCZOS)
        summ = []
        for name in STRATEGIES:
            t, tot = tq[name], tq[name].totals()
            ev = f'  evidence: "{t.evidence}"' if t.evidence else ""
            summ.append(f"{name:15s} {'found' if t.found else 'not found':9s} {'correct' if t.meta['correct'] else 'WRONG':7s} "
                        f"calls {tot['n_calls']:2d}  pixels {100 * tot['pixels_sampled_frac']:5.1f}%  "
                        f"vision tok {tot['vision_tokens']:4d}  out tok {tot['completion_tokens']:3d}  "
                        f"{1000 * tot['time_to_answer_s']:5.0f} ms{ev}")
        img = render(frame, tq, {**q}, thumb=thumb, summary_lines=summ)
        save_png(img, REPO_ROOT / f"results/step4_trace_{qid}.png")

    summary = {"queries": [q["query_id"] for q in queries], "table_md": "\n".join(lines),
               "per_call_type_md": "\n".join(role_lines), "field_costs": costs, "token_check": tokens,
               "thumbnail_identity_multi_fixation_searches": len(multi),
               "eye_repeats": sum(t.repeats for t in eye), "byte_identity_dev_sweep": sweep, "prompts_version": cfg["prompts"]["version"],
               "thumbnail_size_example": list(thumbnail_size(640, 480, int(cfg.get_key("search.thumbnail_long_side"))))}
    (REPO_ROOT / "results/step4_check.json").write_text(json.dumps(summary, indent=1, default=str) + "\n")
    print("wrote results/step4_traces.jsonl, results/step4_check.json, results/step4_trace_<id>.png")


if __name__ == "__main__":
    asyncio.run(main())
