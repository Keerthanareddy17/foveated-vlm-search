#!/usr/bin/env python
"""Step 8 report outputs (no GPU):
  results/step8_sweep_ft.csv, results/step8_sweep_base_vs_ft.png   thumbnail sweep, base vs fine-tuned (test)
  results/step8_val_threeway.md / .json                            val samples: base BF16 vs fine-tuned BF16 vs FP8
  results/step8_throughput.png                                     end-to-end queries/s vs concurrency
"""

from __future__ import annotations

import csv
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from foveated_search.config import REPO_ROOT  # noqa: E402
from foveated_search.eval.metrics import summarize  # noqa: E402
from foveated_search.eval.runner import eval_config  # noqa: E402
from foveated_search.eval.splits import split_ids  # noqa: E402
from foveated_search.eval.table import load_run  # noqa: E402

# validated reference palette (dataviz skill); aqua/yellow are below 3:1 contrast -> direct labels on every line
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0", "#fcfcfb"


def style(ax, fig):
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    for sp in ["left", "bottom"]:
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK2)


def sweep() -> None:
    cfg = eval_config()
    test = split_ids(cfg, "test")
    h = cfg["harness"]

    def s(run):
        tr = [t for t in load_run(run)[0] if t["meta"]["query_id"] in test]
        return summarize(tr, int(h["bootstrap_resamples"]), float(h["ci_level"]), int(cfg["seed"]))

    series = {"base": [(64, "sweep_thumb64"), (96, "R4_eye_search"), (128, "sweep_thumb128"), (192, "sweep_thumb192")],
              "fine-tuned": [(64, "sweep_ft_thumb64"), (96, "R5_eye_search"), (128, "sweep_ft_thumb128"), (192, "sweep_ft_thumb192")]}
    rows = []
    for model, pts in series.items():
        for L, run in pts:
            x = s(run)
            rows.append({"model": model, "thumbnail_long_side": L, "crop": 224, "run": run, "n": x["n"],
                         "found": round(x["found"], 4), "found_ci_lo": round(x["found_ci"][0], 4),
                         "found_ci_hi": round(x["found_ci"][1], 4), "fp_rate": round(x["fp_rate"], 4),
                         "accuracy": round(x["accuracy"], 4), "wrong_location": round(x["wrong_location"], 4),
                         "pixels_frac": round(x["pixels_frac"], 4), "vision_tokens": round(x["vision_tokens"], 1),
                         "output_tokens": round(x["completion_tokens"], 1), "calls": round(x["calls"], 3),
                         "fixations": round(x["fixations"], 3)})
    refs = {"R1 full frame (base)": s("R1_full_frame"), "R1-ft full frame (fine-tuned)": s("R1ft_full_frame")}
    with open(REPO_ROOT / "results/step8_sweep_ft.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5})
    fig, ax = plt.subplots(figsize=(8.4, 5.2), dpi=150)
    style(ax, fig)
    for model, color in [("base", BLUE), ("fine-tuned", ORANGE)]:
        pts = [r for r in rows if r["model"] == model]
        x = [100 * r["pixels_frac"] for r in pts]
        y = [100 * r["found"] for r in pts]
        ax.errorbar(x, y, yerr=[[100 * (r["found"] - r["found_ci_lo"]) for r in pts],
                                [100 * (r["found_ci_hi"] - r["found"]) for r in pts]],
                    fmt="none", ecolor=color, elinewidth=1, alpha=0.45, zorder=2)
        ax.plot(x, y, "-", color=color, lw=2, zorder=3, label=f"eye search, {model} model (crop 224)")
        ax.scatter(x, y, s=46, color=color, edgecolor=SURFACE, linewidth=2, zorder=4)
        for r, xi, yi in zip(pts, x, y):  # fine-tuned labels above-left, base labels below-right
            ax.annotate(f"{r['thumbnail_long_side']} px", (xi, yi), textcoords="offset points",
                        xytext=(8, -13) if model == "base" else (-8, 7), ha="left" if model == "base" else "right",
                        color=INK2, fontsize=8.5)
    for (label, x), off in zip(refs.items(), [(-12, -26), (-12, 8)]):
        px, py = 100 * x["pixels_frac"], 100 * x["found"]
        ax.scatter([px], [py], s=60, facecolor=SURFACE, edgecolor=INK2, linewidth=1.8, marker="D", zorder=4)
        ax.annotate(f"{label}\nFP {100 * x['fp_rate']:.1f}% on negatives", (px, py), textcoords="offset points",
                    xytext=off, ha="right", color=INK2, fontsize=8.5)
    ax.scatter([], [], s=60, facecolor=SURFACE, edgecolor=INK2, linewidth=1.8, marker="D", label="full-frame references")
    ax.set_xscale("log")  # eye-search settings sit at 15-30% of the frame, the references at 100%
    ax.set_xticks([10, 15, 20, 30, 50, 100])
    ax.set_xticklabels(["10", "15", "20", "30", "50", "100"])
    ax.minorticks_off()
    ax.set_xlim(12, 125)
    ax.set_ylim(55, 100)
    ax.set_xlabel("mean pixels sampled per query (% of frame, log scale)", color=INK2)
    ax.set_ylabel("found % on positives (95% CI)", color=INK2)
    ax.set_title("Thumbnail-size sweep, base vs fine-tuned Qwen3-VL-2B (COCO test, BF16)", loc="left", color=INK,
                 fontsize=12, pad=12)
    leg = ax.legend(loc="lower right", frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color(INK2)
    fps = [100 * r["fp_rate"] for r in rows]
    fig.text(0.01, 0.01, f"Eye-search FP on negatives across all 8 settings: {min(fps):.1f}-{max(fps):.1f}% "
             "(n = 90 negatives, 360 positives).\nData: results/step8_sweep_ft.csv, results/step5_sweeps.csv.",
             color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(REPO_ROOT / "results/step8_sweep_base_vs_ft.png", facecolor=SURFACE)
    for r in rows:
        print(r["model"], r["thumbnail_long_side"], r["found"], r["fp_rate"], r["accuracy"], r["pixels_frac"])


def val_threeway() -> None:
    files = {"base BF16": "results/step6_base_on_val.json", "fine-tuned BF16": "results/step7_finetuned_on_val.json",
             "fine-tuned FP8": "results/step8_fp8_on_val.json"}
    data = {k: json.loads((REPO_ROOT / v).read_text()) for k, v in files.items()}
    ids = [r["id"] for r in data["base BF16"]["rows"]]
    assert all([r["id"] for r in d["rows"]] == ids for d in data.values()), "different val samples"
    subsets = [("first glance", "where", "pos", "positives"), ("first glance", "where", "neg", "negatives (absent)"),
               ("look again", "look_again", "pos", "positives"), ("look again", "look_again", "neg", "negatives (absent)"),
               ("confirm", "confirm", "yes", "yes crops"), ("confirm", "confirm", "no", "no crops (all)")]
    kinds = [("confirm", "near_miss"), ("confirm", "distractor"), ("confirm", "background")]
    L = ["# Val samples: base vs fine-tuned vs FP8 (same 900 samples, Step 6 scorer)", "",
         "| skill | subset | n | " + " | ".join(files) + " |", "|---|---|---|" + "---|" * len(files)]
    out = {}

    def rate(rows, cond):
        rs = [r for r in rows if cond(r)]
        return sum(r["ok"] for r in rs) / len(rs), len(rs)

    for skill, t, pol, name in subsets:
        vals = {k: rate(d["rows"], lambda r: r["type"] == t and r["polarity"] == pol) for k, d in data.items()}
        n = next(iter(vals.values()))[1]
        out[f"{t}/{pol}"] = {k: v[0] for k, v in vals.items()}
        L.append(f"| {skill} | {name} | {n} | " + " | ".join(f"{100 * v[0]:.1f}" for v in vals.values()) + " |")
    for t, k in kinds:
        vals = {m: rate(d["rows"], lambda r: r["type"] == t and r["no_kind"] == k) for m, d in data.items()}
        n = next(iter(vals.values()))[1]
        out[f"{t}/no_{k}"] = {m: v[0] for m, v in vals.items()}
        L.append(f"| confirm | no: {k.replace('_', ' ')} | {n} | " + " | ".join(f"{100 * v[0]:.1f}" for v in vals.values()) + " |")
    L += ["", "Unparsed answers: " + ", ".join(f"{k}: {sum(v['unparsed'] for v in d['summary'].values())}" for k, d in data.items())]
    (REPO_ROOT / "results/step8_val_threeway.md").write_text("\n".join(L) + "\n")
    (REPO_ROOT / "results/step8_val_threeway.json").write_text(json.dumps(out, indent=1) + "\n")
    print("\n".join(L))


def throughput() -> None:
    rows = [r for r in csv.DictReader(open(REPO_ROOT / "results/step8_serving.csv")) if r["kind"] == "e2e"]
    series = [("R1ft", "full frame, fine-tuned BF16 (R1-ft)", BLUE, "o"),
              ("R5", "eye search, fine-tuned BF16 (R5)", ORANGE, "s"),
              ("R6", "eye search, fine-tuned FP8 (R6)", AQUA, "^"),
              ("R1ft-fp8", "full frame, fine-tuned FP8 (reference)", YELLOW, "D")]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5})
    fig, ax = plt.subplots(figsize=(8.4, 5.0), dpi=150)
    style(ax, fig)
    ends = []
    for name, label, color, marker in series:
        pts = sorted(((int(r["concurrency"]), float(r["queries_per_s"])) for r in rows if r["name"] == name))
        if not pts:
            continue
        ax.plot([c for c, _ in pts], [q for _, q in pts], "-", color=color, lw=2, marker=marker, ms=7, mec=SURFACE,
                mew=1.5, label=label)
        ends.append(pts[-1])
    placed = []  # end-of-line value labels, nudged apart so they never overlap
    for c, q in sorted(ends, key=lambda e: -e[1]):
        y = q
        while any(abs(y - p) < 0.55 for p in placed):
            y -= 0.1
        placed.append(y)
        ax.annotate(f"{q:.1f}", (c, q), xytext=(c * 1.06, y), textcoords="data", va="center", color=INK2, fontsize=9)
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 4, 8, 16])
    ax.set_xticklabels(["1", "4", "8", "16"])
    ax.set_xlim(0.8, 22)
    ax.set_xlabel("concurrent searches (one L4, vLLM)", color=INK2)
    ax.set_ylabel("completed searches per second", color=INK2)
    ax.set_title("End-to-end search throughput per L4 (100-query latency subset, 640x480 COCO frames)", loc="left",
                 color=INK, fontsize=11.5, pad=12)
    leg = ax.legend(loc="upper left", frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color(INK2)
    fig.text(0.01, 0.01, "Server on the same VM; image encoding off the event loop; CUDA graphs on, prefix caching off. "
             "Data: results/step8_serving.csv.", color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(REPO_ROOT / "results/step8_throughput.png", facecolor=SURFACE)
    print("wrote results/step8_throughput.png")


if __name__ == "__main__":
    sweep()
    val_threeway()
    throughput()
