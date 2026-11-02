#!/usr/bin/env python
"""Step 10: every number in README.md, computed from results/ and rendered into the README template.

  python scripts/build_readme_tables.py            (make readme-tables)  -> README.md + results/readme_facts.json
  python scripts/build_readme_tables.py --check    also list every digit in README.md that did not come from a fact

The prose lives in scripts/readme_template.md with {{fact}} and {{table:name}} placeholders. Facts are computed here
from results/ with the same code as results/table.md (eval/metrics.py: bootstrap CIs, paired comparisons, seed 42),
so README numbers can't drift from the measurements. Each fact records its source file in readme_facts.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from pathlib import Path

from foveated_search.config import REPO_ROOT
from foveated_search.eval.metrics import latency, paired_compare, second_fixation, summarize
from foveated_search.eval.runner import eval_config, run_specs
from foveated_search.eval.splits import split_ids
from foveated_search.eval.table import load_run

R = REPO_ROOT / "results"
TEMPLATE = REPO_ROOT / "scripts" / "readme_template.md"
OUT = REPO_ROOT / "README.md"
FACTS_OUT = R / "readme_facts.json"


class Facts:
    def __init__(self):
        self.v: dict[str, str] = {}
        self.src: dict[str, str] = {}

    def add(self, key: str, value: str, source: str) -> str:
        if key in self.v and self.v[key] != value:
            raise ValueError(f"fact {key} set twice: {self.v[key]} vs {value}")
        self.v[key], self.src[key] = str(value), source
        return str(value)


def pct(x: float, d: int = 1) -> str:
    return f"{100 * x:.{d}f}"


def ci(lo: float, hi: float) -> str:
    return f"[{pct(lo)}, {pct(hi)}]"


def sci(lo: float, hi: float) -> str:
    return f"[{100 * lo:+.1f}, {100 * hi:+.1f}]"


def ratio(a: float, b: float, d: int = 1) -> str:
    return f"{a / b:.{d}f}"


def md_table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


# ---------------------------------------------------------------- benchmark (R1-R7 + reference rows)
ROW_META = {  # model, precision, strategy (labels only; numbers come from the runs)
    "R1": ("base", "BF16", "full frame"), "R2": ("base", "BF16", "sliding window"),
    "R3": ("base", "BF16", "uniform downscale"), "R4": ("base", "BF16", "eye search"),
    "R5": ("fine-tuned", "BF16", "eye search"), "R6": ("fine-tuned", "FP8", "eye search"),
    "R7": ("fine-tuned", "FP8 + prefix caching", "eye search"),
    "R1-ft": ("fine-tuned", "BF16", "full frame"), "R3-ft": ("fine-tuned", "BF16", "uniform downscale"),
    "R4-192": ("base", "BF16", "eye search, 192 px thumbnail"),
}


def benchmark(F: Facts, T: dict, cfg) -> dict:
    spec = run_specs()
    h = cfg["harness"]
    test = split_ids(cfg, "test")
    rows = {**{k: v for k, v in spec["table"].items()}, **{k: v for k, v in spec["reference"].items()}}
    out = {}
    for rid, r in rows.items():
        acc = [t for t in load_run(r["acc"])[0] if t["meta"]["query_id"] in test]
        s = summarize(acc, int(h["bootstrap_resamples"]), float(h["ci_level"]), int(cfg["seed"]))
        lat = latency(load_run(r["lat"])[0]) if "lat" in r else None
        src = f"results/runs/{r['acc']}.jsonl" + (f" + {r['lat']}.jsonl" if "lat" in r else "") + " (test split)"
        k = rid
        F.add(f"{k}.n", str(s["n"]), src)
        F.add(f"{k}.found", pct(s["found"]), src)
        F.add(f"{k}.found_ci", ci(*s["found_ci"]), src)
        F.add(f"{k}.fp", pct(s["fp_rate"]), src)
        F.add(f"{k}.fp_ci", ci(*s["fp_ci"]), src)
        F.add(f"{k}.acc", pct(s["accuracy"]), src)
        F.add(f"{k}.acc_ci", ci(*s["accuracy_ci"]), src)
        F.add(f"{k}.pix", pct(s["pixels_frac"]), src)
        F.add(f"{k}.tok", f"{s['vision_tokens']:.0f}", src)
        F.add(f"{k}.fix", f"{s['fixations']:.2f}", src)
        F.add(f"{k}.calls", f"{s['calls']:.2f}", src)
        F.add(f"{k}.out_tok", f"{s['completion_tokens']:.0f}", src)
        if lat:
            F.add(f"{k}.lat_n", str(lat["n"]), src)
        if lat:
            F.add(f"{k}.p50", f"{1000 * lat['tta_p50']:.0f}", src)
            F.add(f"{k}.p95", f"{1000 * lat['tta_p95']:.0f}", src)
        out[rid] = {"s": s, "lat": lat, "acc_run": r["acc"]}
    F.add("test.n", str(out["R1"]["s"]["n"]), "results/eval_manifest.jsonl via eval/splits.py")
    F.add("test.n_pos", str(out["R1"]["s"]["n_pos"]), "test split")
    F.add("test.n_neg", str(out["R1"]["s"]["n_neg"]), "test split")

    def trow(rid):
        m = ROW_META[rid]
        lat = f"{F.v[rid + '.p50']} / {F.v[rid + '.p95']}" if rid + ".p50" in F.v else "–"
        return [f"**{rid}**", m[0], m[1], m[2], f"{F.v[rid + '.found']} {F.v[rid + '.found_ci']}",
                f"{F.v[rid + '.fp']} {F.v[rid + '.fp_ci']}", f"**{F.v[rid + '.acc']}** {F.v[rid + '.acc_ci']}",
                F.v[rid + ".fix"] if "eye search" in m[2] else "–", F.v[rid + ".pix"], F.v[rid + ".tok"], lat]
    hdr = ["row", "model", "precision", "strategy", "found % [95% CI]", "false positives % [95% CI]",
           "accuracy % [95% CI]", "fixations / search", "pixels sampled %", "vision tokens / search",
           "time to answer p50 / p95 (ms)"]
    T["main"] = md_table(hdr, [trow(r) for r in ["R1", "R2", "R3", "R4", "R5", "R6", "R7"]])
    T["reference"] = md_table(hdr, [trow(r) for r in ["R1-ft", "R3-ft", "R4-192"]])

    # paired comparisons (same test queries, b minus a)
    def paired(key, a, b, metric):
        ta = [t for t in load_run(a)[0] if t["meta"]["query_id"] in test]
        tb = [t for t in load_run(b)[0] if t["meta"]["query_id"] in test]
        c = paired_compare(ta, tb, metric, int(h["bootstrap_resamples"]), float(h["ci_level"]), int(cfg["seed"]))
        src = f"results/runs/{a}.jsonl vs {b}.jsonl (paired bootstrap, test)"
        F.add(f"{key}.diff", f"{100 * c['diff']:+.1f}", src)
        F.add(f"{key}.ci", sci(*c["diff_ci"]), src)
        F.add(f"{key}.a", pct(c["a"]), src)
        F.add(f"{key}.b", pct(c["b"]), src)
        F.add(f"{key}.w2r", str(c["wrong_to_right"]), src)
        F.add(f"{key}.r2w", str(c["right_to_wrong"]), src)
    paired("ft_acc", "R4_eye_search", "R5_eye_search", "accuracy")
    paired("ft_found", "R4_eye_search", "R5_eye_search", "found")
    paired("ffvseye_acc", "R1ft_full_frame", "R5_eye_search", "accuracy")
    paired("fp8_acc", "R5_eye_search", "R6_eye_search", "accuracy")
    paired("prefix_acc", "R6_eye_search", "R7_eye_search", "accuracy")
    paired("mask_acc", "nomask_eye96", "mask_eye96", "accuracy")
    paired("mask_found", "nomask_eye96", "mask_eye96", "found")
    paired("maskvsR6_acc", "R6_eye_search", "mask_eye96", "accuracy")

    for run, key in [("nomask_eye96", "nomask"), ("mask_eye96", "mask"), ("R4_eye_search", "R4sf"),
                     ("R5_eye_search", "R5sf")]:
        x = second_fixation([t for t in load_run(run)[0] if t["meta"]["query_id"] in test])
        src = f"results/runs/{run}.jsonl (second-fixation analysis, test)"
        F.add(f"{key}.repeats", str(x["repeats"]), src)
        F.add(f"{key}.second", str(x["took_second_fixation"]), src)
        F.add(f"{key}.rejected", str(x["with_rejected_crop"]), src)
        F.add(f"{key}.found_after_second", str(x["found_after_second_fixation"]), src)

    # FP8 latency gain and other derived single numbers
    r5, r6 = out["R5"]["lat"]["tta_p50"], out["R6"]["lat"]["tta_p50"]
    F.add("fp8.speedup_pct", f"{100 * (1 - r6 / r5):.0f}", "R5/R6 latency runs (p50 time to answer)")
    F.add("eye_vs_ff.tok_ratio_ft", ratio(out["R1-ft"]["s"]["vision_tokens"], out["R5"]["s"]["vision_tokens"]),
          "R1ft_full_frame vs R5_eye_search vision tokens (test)")

    # refined point on/off (test) and dev decisions
    for run, key in [("R4_refine_off", "refine_off")]:
        s = summarize([t for t in load_run(run)[0] if t["meta"]["query_id"] in test],
                      int(h["bootstrap_resamples"]), float(h["ci_level"]), int(cfg["seed"]))
        F.add(f"{key}.found", pct(s["found"]), f"results/runs/{run}.jsonl (test)")
        F.add(f"{key}.wrongloc", pct(s["wrong_location"]), f"results/runs/{run}.jsonl (test)")
    F.add("R4.wrongloc", pct(out["R4"]["s"]["wrong_location"]), "results/runs/R4_eye_search.jsonl (test)")
    for run, key in [("dev_sw_point", "sw_point"), ("dev_sw_confirm", "sw_confirm")]:
        s = summarize(load_run(run)[0], int(h["bootstrap_resamples"]), float(h["ci_level"]), int(cfg["seed"]))
        F.add(f"{key}.fp", pct(s["fp_rate"], 0), f"results/runs/{run}.jsonl (dev, 50 queries)")
        F.add(f"{key}.found", pct(s["found"]), f"results/runs/{run}.jsonl (dev, 50 queries)")
    return out


# ---------------------------------------------------------------- Step 2/4 findings
def early_findings(F: Facts) -> None:
    pf = json.loads((R / "step4_point_format.json").read_text())
    src = "results/step4_point_format.json (dev, 50 queries)"
    F.add("fmt.p_hit", pct(pf["p"]["hit_rate"]), src)
    F.add("fmt.point2d_hit", pct(pf["point_2d | absent"]["hit_rate"]), src)
    F.add("fmt.gain_pp", f"{100 * (pf['point_2d | absent']['hit_rate'] - pf['p']['hit_rate']):.0f}", src)
    F.add("fmt.p_slope", f"{pf['p']['fit_slope_intercept']['x'][0]:.2f}", src)
    F.add("fmt.point2d_slope", f"{pf['point_2d | absent']['fit_slope_intercept']['x'][0]:.2f}", src)
    fm = json.loads((R / "step2_format.json").read_text())
    src = "results/step2_format.json"
    F.add("fmt.prompt_parse", pct(fm["prompt_json"]["parse_rate"], 0), src)
    F.add("fmt.schema_parse", pct(fm["structured_json_schema"]["parse_rate"], 0), src)


# ---------------------------------------------------------------- Step 6/7: training
def training(F: Facts, T: dict) -> None:
    st = (R / "step6_dataset_stats.md").read_text()
    m = re.search(r"\| samples \| (\d+) \(train (\d+), val (\d+)\)", st)
    src = "results/step6_dataset_stats.md"
    F.add("data.samples", f"{int(m.group(1)):,}", src)
    F.add("data.train", f"{int(m.group(2)):,}", src)
    F.add("data.val", f"{int(m.group(3)):,}", src)
    for t in ["where", "look_again", "confirm"]:
        n = re.search(rf"\| {t} \| (\d+) \(", st)
        F.add(f"data.{t}", f"{int(n.group(1)):,}", src)
    img = re.search(r"\| images \| (\d+) \(train (\d+), val (\d+), shared (\d+)\)", st)
    F.add("data.images", f"{int(img.group(1)):,}", src)
    F.add("data.shared", img.group(4), src)

    log = list(csv.DictReader(open(R / "step7_train_log.csv")))
    src = "results/step7_train_log.csv"
    ev = [r for r in log if r["eval_loss"]]
    F.add("train.steps", log[-1]["step"], src)
    F.add("train.minutes", f"{float(log[-1]['time_s']) / 60:.1f}", src)
    F.add("train.val_loss_final", f"{float(ev[-1]['eval_loss']):.3f}", src)
    F.add("train.val_loss_first", f"{float(ev[0]['eval_loss']):.3f}", src)
    F.add("train.peak_mem_gb", f"{max(float(r['peak_mem_gb']) for r in log):.1f}", src)

    ck = json.loads((R / "step7_checks.json").read_text())["val"]
    src = "results/step7_checks.json (900 val samples, BF16)"
    rows = []
    labels = {"where/positives": ("where to look (first glance)", "target present: points at it"),
              "where/negatives (absent)": ("where to look (first glance)", "target absent: says absent"),
              "look_again/positives": ("look again (after a rejected crop)", "target elsewhere: points at it"),
              "look_again/negatives (absent)": ("look again (after a rejected crop)", "target absent: says absent"),
              "confirm/yes crops": ("confirm (sharp crop)", "crop shows the target: yes"),
              "confirm/no crops": ("confirm (sharp crop)", "crop doesn't: no")}
    for k, (skill, case) in labels.items():
        b, f = pct(ck[k]["base"]), pct(ck[k]["finetuned"])
        key = "val." + k.split("/")[0] + "." + ("pos" if "positives" in k or "yes" in k else "neg")
        F.add(key + ".base", b, src)
        F.add(key + ".ft", f, src)
        rows.append([skill, case, str(ck[k]["n"]), b, f"**{f}**"])
    T["val"] = md_table(["skill", "case", "val samples", "base %", "fine-tuned %"], rows)


# ---------------------------------------------------------------- Step 8: FP8, prefix caching, serving
def serving(F: Facts, T: dict) -> None:
    q = json.loads((R / "step8_fp8_quantize.json").read_text())
    F.add("fp8.ckpt_gb", f"{q['safetensors_gb']:.2f}", "results/step8_fp8_quantize.json")
    F.add("fp8.bf16_modules", str(len(q["quantized_layers_ignore"])), "results/step8_fp8_quantize.json")
    F.add("fp8.quant_s", f"{q['load_s'] + q['quantize_s'] + q['save_s']:.0f}", "results/step8_fp8_quantize.json")
    hub = json.loads((R / "step7_hub_push.json").read_text())
    F.add("bf16.ckpt_gb", f"{hub['files']['model.safetensors'] / 1e9:.2f}", "results/step7_hub_push.json")

    rows = list(csv.DictReader(open(R / "step8_serving.csv")))
    src = "results/step8_serving.csv"
    mem = {r["label"]: r for r in rows}
    F.add("fp8.weights_gib", mem["fp8"]["model_mem_gib"], src)
    F.add("bf16.weights_gib", mem["bf16"]["model_mem_gib"], src)
    F.add("fp8.kv_tokens", f"{int(mem['fp8']['kv_cache_tokens']):,}", src)
    F.add("bf16.kv_tokens", f"{int(mem['bf16']['kv_cache_tokens']):,}", src)

    def call(label, name, conc, field):
        return next(r for r in rows if r["label"] == label and r["kind"] == "call" and r["name"] == name
                    and int(r["concurrency"]) == conc)[field]
    for lab in ["bf16", "fp8", "fp8_eager"]:
        for nm in ["thumb", "full"]:
            F.add(f"call.{lab}.{nm}.p50", f"{float(call(lab, nm, 1, 'total_p50_ms')):.0f}", src)
    F.add("call.fp8.thumb.rps8", f"{float(call('fp8', 'thumb', 8, 'req_per_s')):.1f}", src)
    F.add("call.fp8_eager.thumb.rps8", f"{float(call('fp8_eager', 'thumb', 8, 'req_per_s')):.1f}", src)
    F.add("eager.slowdown_thumb", ratio(float(call("fp8_eager", "thumb", 1, "total_p50_ms")),
                                        float(call("fp8", "thumb", 1, "total_p50_ms"))), src)
    F.add("eager.slowdown_full", ratio(float(call("fp8_eager", "full", 1, "total_p50_ms")),
                                       float(call("fp8", "full", 1, "total_p50_ms"))), src)

    e2e = [r for r in rows if r["kind"] == "e2e"]
    names = {("bf16", "R1ft"): "full frame, fine-tuned BF16", ("bf16", "R5"): "eye search, fine-tuned BF16 (R5)",
             ("fp8", "R6"): "eye search, fine-tuned FP8 (R6)", ("fp8", "R1ft-fp8"): "full frame, fine-tuned FP8"}
    concs = sorted({int(r["concurrency"]) for r in e2e})
    trows = []
    for (lab, nm), title in names.items():
        vals = {int(r["concurrency"]): float(r["queries_per_s"]) for r in e2e if r["label"] == lab and r["name"] == nm}
        if vals:
            trows.append([title] + [f"{vals[c]:.2f}" for c in concs])
            for c in concs:
                F.add(f"tput.{lab}.{nm}.c{c}", f"{vals[c]:.2f}", src)
    T["throughput"] = md_table(["completed searches / s (640x480, one L4)"] + [f"concurrency {c}" for c in concs], trows)

    pc = json.loads((R / "step8_prefix_cache.json").read_text())
    src = "results/step8_prefix_cache.json"
    F.add("prefix.hit_rate", pct(pc["server_metrics"]["test_run_cold_cache"]["prefix_hit_rate"]), src)
    F.add("prefix.fix23_hit", pct(pc["per_call_R7_test"]["thumbnail point, fixations 2-3"]["token_hit_rate"], 0), src)
    F.add("prefix.contaminated", pct(pc["server_metrics"]["latency_run_after_test_run_CONTAMINATED"]["prefix_hit_rate"]), src)
    l = pc["latency_R6_vs_R7_conc1"]
    F.add("prefix.R6.p50", f"{1000 * l['R6']['tta_p50']:.0f}", src)
    F.add("prefix.R7.p50", f"{1000 * l['R7']['tta_p50']:.0f}", src)

    T["fp8"] = md_table(
        ["", "fine-tuned BF16 (R5)", "fine-tuned FP8 (R6)"],
        [["eye-search accuracy, test % [95% CI]", f"{F.v['R5.acc']} {F.v['R5.acc_ci']}", f"{F.v['R6.acc']} {F.v['R6.acc_ci']}"],
         ["paired accuracy change, FP8 minus BF16 (pp)", "", f"{F.v['fp8_acc.diff']} {F.v['fp8_acc.ci']}"],
         ["time to answer p50 / p95 (ms)", f"{F.v['R5.p50']} / {F.v['R5.p95']}", f"{F.v['R6.p50']} / {F.v['R6.p95']}"],
         ["one call, thumbnail / full frame, p50 (ms)", f"{F.v['call.bf16.thumb.p50']} / {F.v['call.bf16.full.p50']}",
          f"{F.v['call.fp8.thumb.p50']} / {F.v['call.fp8.full.p50']}"],
         ["weights in vLLM (GiB)", F.v["bf16.weights_gib"], F.v["fp8.weights_gib"]],
         ["KV-cache capacity (tokens)", F.v["bf16.kv_tokens"], F.v["fp8.kv_tokens"]]])


# ---------------------------------------------------------------- Step 8b: frame-size scaling
def scaling(F: Facts, T: dict) -> list[dict]:
    rows = list(csv.DictReader(open(R / "step8b_scaling.csv")))
    src = "results/step8b_scaling.csv"
    keep = {"full frame": "full", "eye search 288 px": "eye288", "eye search 96 px": "eye96",
            "downscale (eye-96 budget)": "down"}
    out = []
    for r in rows:
        if r["strategy"] not in keep:
            continue
        g, s = r["grid"], keep[r["strategy"]]
        k = f"sc{g}.{s}"
        F.add(k + ".acc", pct(float(r["accuracy"])), src)
        F.add(k + ".acc_ci", ci(float(r["accuracy_ci_lo"]), float(r["accuracy_ci_hi"])), src)
        F.add(k + ".fp", pct(float(r["fp_rate"])), src)
        F.add(k + ".tok", f"{float(r['vision_tokens']):,.0f}", src)
        F.add(k + ".pix", pct(float(r["pixels_frac"])), src)
        F.add(k + ".p50", f"{float(r['tta_p50_ms']):,.0f}", src)
        F.add(k + ".tput", f"{float(r['searches_per_s_c16']):.1f}", src)
        out.append(r)
    tp = list(csv.DictReader(open(R / "step8b_throughput.csv")))
    F.add("tput.n", tp[0]["n"], "results/step8b_throughput.csv")
    F.add("tput.conc", tp[0]["concurrency"], "results/step8b_throughput.csv")
    for g in ["1", "2", "3"]:
        e = next(r for r in rows if r["grid"] == g and r["strategy"] == "eye search 288 px")
        f = next(r for r in rows if r["grid"] == g and r["strategy"] == "full frame")
        F.add(f"sc{g}.tput_ratio", ratio(float(e["searches_per_s_c16"]), float(f["searches_per_s_c16"])), src)
        F.add(f"sc{g}.frame", e["frame"].split("(")[1].rstrip(")"), src)
        F.add(f"sc{g}.n", e["n"], src)
    order = ["full frame", "eye search 288 px", "eye search 96 px", "downscale (eye-96 budget)"]
    trows = []
    for g in ["1", "2", "3"]:
        for name in order:
            r = next((x for x in rows if x["grid"] == g and x["strategy"] == name), None)
            if r is None:
                continue
            k = f"sc{g}.{keep[name]}"
            trows.append([r["frame"] if name == "full frame" else "", name, f"{F.v[k + '.acc']} {F.v[k + '.acc_ci']}",
                          F.v[k + ".fp"], F.v[k + ".tok"], F.v[k + ".pix"], F.v[k + ".p50"], F.v[k + ".tput"]])
    T["scaling"] = md_table(["frame", "strategy", "accuracy % [95% CI]", "false positives %", "vision tokens / search",
                             "pixels sampled %", "time to answer p50 (ms)", "searches / s at concurrency 16"], trows)
    return out


# ---------------------------------------------------------------- Step 9: photo demo + sweep
def photo_name(spec: dict, image: str) -> str:
    """Plain photo name from configs/demo_queries.yaml `photo_names` (falls back to the file name)."""
    return (spec.get("photo_names") or {}).get(Path(image).name, Path(image).name)


def example(F: Facts, spec: dict) -> None:
    """The step-by-step example: the replayed (median-time) eye-search trace of desk/apple-near, "food"."""
    rec = json.loads((R / "demo" / "positive2__food.json").read_text())
    src = "results/demo/positive2__food.json (eye-search trace shown in the replay)"
    tr, sm = rec["traces"]["eye_search"], rec["summary"]["eye_search"]
    pt = next(c for c in tr["calls"] if c["role"] == "point")
    cf = next(c for c in tr["calls"] if c["role"] == "confirm")
    F.add("ex.photo", photo_name(spec, rec["image"]), "configs/demo_queries.yaml photo_names")
    F.add("ex.thumb", f"{pt['sent_w']}x{pt['sent_h']}", src)
    F.add("ex.thumb_tok", str(pt["vision_tokens"]), src)
    F.add("ex.thumb_kb", f"{pt['png_bytes'] / 1e3:.0f}", src)
    F.add("ex.thumb_ms", f"{1000 * pt['total_s']:.0f}", src)
    F.add("ex.point_answer", pt["raw_answer"], src)
    fx, fy = tr["fixations"][0]
    F.add("ex.fix", f"({fx:.0f}, {fy:.0f})", src)
    x0, y0, x1, y1 = cf["region"]
    F.add("ex.crop", f"{cf['sent_w']}x{cf['sent_h']}", src)
    F.add("ex.crop_box", f"({x0}, {y0})-({x1}, {y1})", src)
    F.add("ex.crop_tok", str(cf["vision_tokens"]), src)
    F.add("ex.crop_kb", f"{cf['png_bytes'] / 1e3:.0f}", src)
    F.add("ex.crop_ms", f"{1000 * cf['total_s']:.0f}", src)
    F.add("ex.confirm_answer", cf["raw_answer"], src)
    F.add("ex.label", tr["evidence"], src)
    fp = tr["final_point"]
    F.add("ex.final", f"({fp[0]:.0f}, {fp[1]:.0f})", src)
    F.add("ex.total_s", f"{sm['time_to_answer_s']:.2f}", src)
    F.add("ex.tok", str(sm["vision_tokens"]), src)
    F.add("ex.pix", f"{100 * sm['pixels_sent'] / (rec['frame_size'][0] * rec['frame_size'][1]):.1f}", src)
    F.add("ex.pix_round", f"{100 * sm['pixels_sent'] / (rec['frame_size'][0] * rec['frame_size'][1]):.0f}", src)
    F.add("ex.crop_side", str(rec["settings"]["fixation_size"]), src)


def demo(F: Facts, T: dict) -> None:
    import yaml
    spec = yaml.safe_load((REPO_ROOT / "configs/demo_queries.yaml").read_text())
    from foveated_search.demo.run_demo import answer_note, notes_for
    example(F, spec)
    for img, nm in (spec.get("photo_names") or {}).items():
        F.add(f"photo.{Path(img).stem}", nm, "configs/demo_queries.yaml photo_names")
    trows = []
    for q in spec["runs"]:
        name = Path(q["image"]).stem
        p = R / "demo" / f"{name}__{q['query']}.json"
        rec = json.loads(p.read_text())
        src = f"results/demo/{p.name}"
        k = f"demo.{name}.{q['query']}"
        F.add(k + ".frame", f"{rec['frame_size'][0]}x{rec['frame_size'][1]}", src)
        F.add(k + ".thumb", str(rec["settings"]["thumbnail_long_side"]), src)
        for s in ["full_frame", "eye_search"]:
            sm = rec["summary"][s]
            ks = f"{k}.{'ff' if s == 'full_frame' else 'eye'}"
            F.add(ks + ".t", f"{sm['time_median_s']:.2f}", src)
            F.add(ks + ".tok", f"{sm['vision_tokens']:,}", src)
            F.add(ks + ".kb", f"{sm['png_bytes'] / 1e3:,.0f}", src)
            F.add(ks + ".pix", f"{100 * sm['pixels_sent'] / (rec['frame_size'][0] * rec['frame_size'][1]):.1f}", src)
            F.add(ks + ".correct", f"{sm['n_correct']}/{sm['repeats']}", src)
            note = answer_note(sm, rec["query"], notes_for(spec, rec["image"]))
            verdict = ("correct" if sm["correct"] else "**wrong**") + (f" ({note})" if note else "")
            looks = "–" if s == "full_frame" else str(sm["fixations"])
            trows.append([f"{photo_name(spec, rec['image'])} · \"{rec['query']}\"" if s == "full_frame" else "",
                          ("absent" if rec.get("expect") == "absent" else "present") if s == "full_frame" else "",
                          s.replace("_", " "), sm["status"], verdict, looks, F.v[ks + ".pix"],
                          f"{sm['png_bytes'] / 1e3:,.0f} KB", f"{sm['vision_tokens']:,}",
                          f"{sm['time_median_s']:.3f} / {sm['time_min_s']:.3f} / {sm['time_max_s']:.3f}"])
        e, f = rec["summary"]["eye_search"], rec["summary"]["full_frame"]
        F.add(k + ".tok_ratio", ratio(f["vision_tokens"], e["vision_tokens"]), src)
        F.add(k + ".bytes_ratio", ratio(f["png_bytes"], e["png_bytes"]), src)
        F.add(k + ".reps", str(e["repeats"]), src)
    T["demo"] = md_table(["photo · query", "target", "strategy", "answer", "correct (all 5 runs)", "looks",
                          "pixels sampled %", "bytes sent (PNG)", "vision tokens", "time median / min / max (s)"], trows)
    n_q = len(spec["runs"])
    F.add("demo.n_queries", str(n_q), "configs/demo_queries.yaml")
    q0 = next(q for q in spec["runs"] if q["image"].endswith("/positive.jpg") and q["query"] == "apple")
    bx = q0["gt_boxes"][0]
    fw, fh = json.loads((R / "demo" / "positive__apple.json").read_text())["frame_size"]
    F.add("demo.apple_pos.box", f"{bx[2] - bx[0]}x{bx[3] - bx[1]}", "configs/demo_queries.yaml (hand-measured box)")
    F.add("demo.apple_pos.share", f"{100 * (bx[2] - bx[0]) * (bx[3] - bx[1]) / (fw * fh):.2f}",
          "configs/demo_queries.yaml box / results/demo/positive__apple.json frame size")
    for s, short in [("full_frame", "ff"), ("eye_search", "eye")]:
        ok = sum(json.loads((R / "demo" / f"{Path(q['image']).stem}__{q['query']}.json").read_text())["summary"][s]["correct"]
                 for q in spec["runs"])
        F.add(f"demo.{short}.correct", f"{ok}/{n_q}", "results/demo/*.json")

    sw = list(csv.DictReader(open(R / "demo/thumbnail_sweep.csv")))
    src = "results/demo/thumbnail_sweep.csv"
    sizes = sorted({r["thumbnail_long_side"] for r in sw if r["strategy"] == "eye_search"}, key=int)
    trows = []
    for sz in sizes:
        rs = [r for r in sw if r["strategy"] == "eye_search" and r["thumbnail_long_side"] == sz]
        ok = sum(r["correct"] == "True" for r in rs)
        apple = {r["image"]: r["target_in_thumb_px"] for r in rs if r["query"] == "apple" and r["target_in_thumb_px"]}
        tok = [int(r["vision_tokens"]) for r in rs]
        F.add(f"sweep.{sz}.correct", f"{ok}/{len(rs)}", src)
        F.add(f"sweep.{sz}.apple_positive", apple.get("positive.jpg", ""), src)
        trows.append([f"{sz} px", f"{ok}/{len(rs)}", f"{apple.get('positive.jpg', '')} / {apple.get('positive2.jpg', '')}",
                      f"{min(tok)}–{max(tok)}",
                      f"{min(float(r['time_median_s']) for r in rs):.2f}–{max(float(r['time_median_s']) for r in rs):.2f}"])
    ff = [r for r in sw if r["strategy"] == "full_frame"]
    trows.append(["full frame (for reference)", f"{sum(r['correct'] == 'True' for r in ff)}/{len(ff)}", "native",
                  f"{min(int(r['vision_tokens']) for r in ff):,}–{max(int(r['vision_tokens']) for r in ff):,}",
                  f"{min(float(r['time_median_s']) for r in ff):.2f}–{max(float(r['time_median_s']) for r in ff):.2f}"])
    T["sweep"] = md_table(["eye-search thumbnail", "correct",
                           f"apple in the thumbnail, {photo_name(spec, 'positive.jpg')} / {photo_name(spec, 'positive2.jpg')} (px)",
                           "vision tokens / search", "time to answer, medians (s)"], trows)
    s3 = {r["thumbnail_long_side"]: r for r in sw if r["strategy"] == "eye_search"}
    F.add("sweep.runs", next(r["repeats"] for r in sw if r["strategy"] == "eye_search"), src)


def settings(F: Facts) -> None:
    """Settings the README states (from configs/ and the frozen manifest; not measurements)."""
    import yaml
    tr = yaml.safe_load((REPO_ROOT / "configs/train.yaml").read_text())
    t = tr[next(iter(tr))] if "lora" not in tr else tr
    src = "configs/train.yaml"
    F.add("cfg.lora_r", str(t["lora"]["r"]), src)
    F.add("cfg.lora_alpha", str(t["lora"]["alpha"]), src)
    F.add("cfg.epochs", str(t["epochs"]), src)
    F.add("cfg.batch", str(int(t["per_device_batch_size"]) * int(t["grad_accum"])), src)
    se = yaml.safe_load((REPO_ROOT / "configs/search.yaml").read_text())["search"]
    src = "configs/search.yaml"
    F.add("cfg.thumb", str(se["thumbnail_long_side"]), src)
    F.add("cfg.crop", str(se["fixation_size"]), src)
    F.add("cfg.max_fix", str(se["max_fixations"]), src)
    rows = [json.loads(line) for line in (R / "eval_manifest.jsonl").read_text().splitlines() if line.strip()]
    src = "results/eval_manifest.jsonl"
    F.add("manifest.n", str(len(rows)), src)
    F.add("manifest.neg", str(sum(r["is_negative"] for r in rows)), src)
    cats = {r["query"] for r in rows if r.get("query_type") == "category"}
    F.add("manifest.n_categories", str(len(cats)), src)
    F.add("manifest.dev", str(sum(1 for _ in (R / "eval_manifest_dev.jsonl").open())), "results/eval_manifest_dev.jsonl")


def environment(F: Facts) -> None:
    env = json.loads((R / "env_report.json").read_text())
    blob = json.dumps(env)
    F.add("env.gpu", "NVIDIA L4", "results/env_report.json")
    mib = re.search(r"(\d{5}) MiB", blob)
    if mib:
        F.add("env.gpu_mib", f"{int(mib.group(1)):,}", "results/env_report.json")


# ---------------------------------------------------------------- render + check
PH = re.compile(r"\{\{\s*(table:)?([A-Za-z0-9_.\-]+)\s*\}\}")


def render(template: str, F: Facts, T: dict, mark: bool = False) -> str:
    def sub(m):
        if m.group(1):
            if m.group(2) not in T:
                raise KeyError(f"unknown table {m.group(2)}")
            val = T[m.group(2)]
        else:
            if m.group(2) not in F.v:
                raise KeyError(f"unknown fact {m.group(2)}")
            val = F.v[m.group(2)]
        return f"\x01{val}\x02" if mark else val
    return PH.sub(sub, template)


def untraced(marked: str) -> list[str]:
    """Digits in the rendered README that did not come from a fact/table, with a little context."""
    plain = re.sub(r"\x01.*?\x02", " ", marked, flags=re.S)
    plain = re.sub(r"```bibtex.*?```", " ", plain, flags=re.S)              # citations, checked separately
    plain = re.sub(r"<!--.*?-->", " ", plain, flags=re.S)
    plain = re.sub(r"\]\([^)]*\)|href=\"[^\"]*\"|src=\"[^\"]*\"|srcset=\"[^\"]*\"", " ", plain)  # link targets
    hits = []
    for line in plain.splitlines():
        for m in re.finditer(r"\d[\d,.]*", line):
            ctx = line[max(0, m.start() - 30): m.end() + 25].strip()
            hits.append(f"{m.group(0):>10s}  …{ctx}…")
    return hits


def compute() -> tuple[Facts, dict]:
    """All facts and tables (also used by scripts/make_figures.py, so figures and README agree)."""
    cfg = eval_config()
    F, T = Facts(), {}
    benchmark(F, T, cfg)
    early_findings(F)
    training(F, T)
    serving(F, T)
    scaling(F, T)
    demo(F, T)
    settings(F)
    environment(F)
    return F, T


def build(check: bool) -> int:
    F, T = compute()
    template = TEMPLATE.read_text()
    OUT.write_text(render(template, F, T))
    FACTS_OUT.write_text(json.dumps({k: {"value": F.v[k], "source": F.src[k]} for k in sorted(F.v)}, indent=1) + "\n")
    used = set(m.group(2) for m in PH.finditer(template) if not m.group(1))
    print(f"README.md rendered: {len(used)} facts used of {len(F.v)} computed, {len(T)} tables "
          f"-> {FACTS_OUT.relative_to(REPO_ROOT)}")
    if check:
        hits = untraced(render(template, F, T, mark=True))
        print(f"\n{len(hits)} digit groups in README.md not produced by a fact (labels, settings, years, names):")
        for h in hits:
            print("  " + h)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true")
    sys.exit(build(ap.parse_args().check))
