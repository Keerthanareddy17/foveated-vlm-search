#!/usr/bin/env python
"""Step 10: README figures, each in a light and a dark variant (docs/assets/<name>_{light,dark}.png).

  python scripts/make_figures.py        (make figures)

Numbers come from scripts/build_readme_tables.py (compute()), i.e. from results/, so figures and README agree.
Style: one surface + ink per theme, three categorical hues validated for both themes (all-pairs CVD and contrast;
aqua is below 3:1 on the light surface, so every series is also direct-labelled), thin marks, recessive grid.
Encoding used everywhere: hue = strategy (full frame / eye search / other), filled = fine-tuned, hollow = base.
"""

from __future__ import annotations

import csv
import io
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_readme_tables import R, compute  # noqa: E402

from foveated_search.config import REPO_ROOT  # noqa: E402
from foveated_search.search.views import make_thumbnail  # noqa: E402

OUT = REPO_ROOT / "docs" / "assets"
THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "grid": "#e4e3df", "axis": "#b9b8b2",
              "ff": "#2a78d6", "eye": "#eb6834", "other": "#1baf7a"},
    "dark": {"surface": "#1a1a19", "ink": "#f0efec", "ink2": "#c3c2b7", "grid": "#33332f", "axis": "#5a5954",
             "ff": "#3987e5", "eye": "#d95926", "other": "#199e70"},
}
DPI = 160


def style(ax, th, xlabel="", ylabel=""):
    ax.set_facecolor(th["surface"])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(th["axis"])
    ax.tick_params(colors=th["ink2"], labelsize=9)
    ax.grid(True, color=th["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel(xlabel, color=th["ink2"], fontsize=10)
    ax.set_ylabel(ylabel, color=th["ink2"], fontsize=10)


def fig(th, w, h, ncols=1):
    f, axes = plt.subplots(1, ncols, figsize=(w, h), dpi=DPI)
    f.patch.set_facecolor(th["surface"])
    return f, axes


def title(f, th, text, sub=None):
    f.text(0.01, 0.985, text, ha="left", va="top", fontsize=12.5, fontweight="bold", color=th["ink"])
    if sub:
        f.text(0.01, 0.925, sub, ha="left", va="top", fontsize=9.5, color=th["ink2"])


def save(f, name, theme):
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{name}_{theme}.png"
    f.savefig(p, facecolor=f.get_facecolor(), dpi=f.dpi)
    plt.close(f)
    return p


def marker(ax, x, y, color, filled, th, size=70, z=3):
    ax.scatter([x], [y], s=size, facecolor=color if filled else th["surface"], edgecolor=color, linewidth=2, zorder=z)


# ---------------------------------------------------------------- 1. hero trade-off
def hero(F, theme):
    th = THEMES[theme]
    f, ax = fig(th, 9.6, 5.6)
    f.subplots_adjust(left=0.08, right=0.98, top=0.83, bottom=0.11)
    pts = [  # row, family, fine-tuned, label, (dx, dy) label offset in points, ha
        ("R1", "ff", False, "R1 base · full frame", (8, -4), "left"),
        ("R2", "other", False, "R2 base · sliding window", (8, -14), "left"),
        ("R3", "other", False, "R3 base · downscale", (-10, -4), "right"),
        ("R4", "eye", False, "R4 base · eye search (96 px)", (10, -2), "left"),
        ("R4-192", "eye", False, "base · eye search, 192 px thumbnail", (10, -3), "left"),
        ("R5", "eye", True, "R5–R7 fine-tuned · eye search\n(BF16 / FP8 / FP8 + prefix cache)", (10, -12), "left"),
        ("R1-ft", "ff", True, "fine-tuned · full frame", (8, 6), "left"),
        ("R3-ft", "other", True, "fine-tuned · downscale", (10, 3), "left"),
    ]
    for rid, fam, ft, label, (dx, dy), ha in pts:
        x, y = float(F.v[f"{rid}.tok"]), float(F.v[f"{rid}.acc"])
        lo, hi = [float(v) for v in F.v[f"{rid}.acc_ci"].strip("[]").split(",")]
        ax.plot([x, x], [lo, hi], color=th[fam], linewidth=1.5, alpha=0.55, zorder=2)
        marker(ax, x, y, th[fam], ft, th, size=80)
        if rid == "R5":  # R6 and R7 sit on the same spot
            for r2 in ("R6", "R7"):
                marker(ax, float(F.v[f"{r2}.tok"]), float(F.v[f"{r2}.acc"]), th[fam], True, th, size=80)
        ax.annotate(label, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va="center",
                    fontsize=9, color=th["ink"])
    ax.set_xscale("log")
    ax.set_xlim(30, 600)
    ax.set_xticks([30, 50, 100, 200, 300, 500])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_ylim(72, 98)
    style(ax, th, "vision tokens per search (log scale)", "accuracy on the 450-query test split (%)")
    # legend for the encoding
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", linestyle="", markersize=8, markerfacecolor=th["ff"], markeredgecolor=th["ff"], label="full frame"),
         Line2D([], [], marker="o", linestyle="", markersize=8, markerfacecolor=th["eye"], markeredgecolor=th["eye"], label="eye search"),
         Line2D([], [], marker="o", linestyle="", markersize=8, markerfacecolor=th["other"], markeredgecolor=th["other"], label="downscale / sliding window"),
         Line2D([], [], marker="o", linestyle="", markersize=8, markerfacecolor=th["surface"], markeredgecolor=th["ink2"], label="base model (hollow)"),
         Line2D([], [], marker="o", linestyle="", markersize=8, markerfacecolor=th["ink2"], markeredgecolor=th["ink2"], label="fine-tuned (filled)")]
    lg = ax.legend(handles=h, loc="lower right", fontsize=8.5, frameon=False, labelcolor=th["ink"])
    title(f, th, "Accuracy vs compute on 640x480 COCO frames (benchmark: 96 px thumbnail)",
          "Vertical bars: 95% bootstrap CI. Accuracy = targets found at the right place + negatives correctly reported absent.")
    return save(f, "hero_tradeoff", theme)


# ---------------------------------------------------------------- 2. frame-size scaling (small multiples)
def scaling_fig(F, theme):
    th = THEMES[theme]
    f, (a1, a2) = fig(th, 10.4, 4.4, 2)
    f.subplots_adjust(left=0.07, right=0.97, top=0.78, bottom=0.14, wspace=0.28)
    xs = [1, 2, 3]
    xl = [F.v[f"sc{g}.frame"] for g in xs]
    series = [("full", "full frame", th["ff"], True), ("eye288", "eye search, 288 px", th["eye"], True),
              ("eye96", "eye search, 96 px", th["eye"], False)]
    for key, label, col, filled in series:
        acc = [float(F.v[f"sc{g}.{key}.acc"]) for g in xs]
        tp = [float(F.v[f"sc{g}.{key}.tput"]) for g in xs]
        for ax, ys in ((a1, acc), (a2, tp)):
            ax.plot(xs, ys, color=col, linewidth=2, linestyle="-" if filled else (0, (4, 3)), zorder=2)
            for x, y in zip(xs, ys):
                marker(ax, x, y, col, filled, th, size=45)
            ax.annotate(label, (xs[-1], ys[-1]), xytext=(8, 0), textcoords="offset points", va="center",
                        fontsize=8.5, color=th["ink"])
    for ax, yl in ((a1, "accuracy (%)"), (a2, "searches / s, concurrency 16")):
        style(ax, th, "frame size (same 308 test queries, as 1x1 / 2x2 / 3x3 mosaics)", yl)
        ax.set_xticks(xs, xl)
        ax.set_xlim(0.8, 3.9)
    a1.set_ylim(35, 100)
    a2.set_ylim(0, 20)
    title(f, th, "Bigger frames: full frame stays most accurate, eye search keeps its cost flat",
          "Fine-tuned 2B, FP8, one L4. Full frame sends the whole frame at native resolution (300 to 2,700 vision tokens).")
    return save(f, "scaling", theme)


# ---------------------------------------------------------------- 3. thumbnail sweep, base vs fine-tuned
def sweep_fig(F, theme):
    th = THEMES[theme]
    rows = list(csv.DictReader(open(R / "step8_sweep_ft.csv")))
    f, ax = fig(th, 7.6, 4.4)
    f.subplots_adjust(left=0.1, right=0.8, top=0.8, bottom=0.14)
    for model, filled in (("base", False), ("fine-tuned", True)):
        rs = sorted([r for r in rows if r["model"] == model], key=lambda r: int(r["thumbnail_long_side"]))
        xs = [int(r["thumbnail_long_side"]) for r in rs]
        ys = [100 * float(r["accuracy"]) for r in rs]
        ax.plot(xs, ys, color=th["eye"], linewidth=2, linestyle="-" if filled else (0, (4, 3)), zorder=2)
        for x, y in zip(xs, ys):
            marker(ax, x, y, th["eye"], filled, th, size=50)
        ax.annotate(f"{model} model", (xs[-1], ys[-1]), xytext=(9, 0), textcoords="offset points", va="center",
                    fontsize=9, color=th["ink"])
    style(ax, th, "eye-search thumbnail long side (px); crop 224 px", "accuracy on the test split (%)")
    ax.set_xticks([64, 96, 128, 192])
    title(f, th, "Fine-tuning helps at every glance size",
          "Eye search, BF16, 450 test queries. Fine-tuned at 64 px roughly matches the base model at 96 px.")
    return save(f, "sweep_base_vs_ft", theme)


# ---------------------------------------------------------------- 4. throughput vs concurrency
def throughput_fig(F, theme):
    th = THEMES[theme]
    rows = [r for r in csv.DictReader(open(R / "step8_serving.csv")) if r["kind"] == "e2e"]
    f, ax = fig(th, 7.6, 4.4)
    f.subplots_adjust(left=0.1, right=0.74, top=0.8, bottom=0.14)
    lines = [("fp8", "R1ft-fp8", "full frame, FP8", th["ff"], "-", 6), ("bf16", "R1ft", "full frame, BF16", th["ff"], (0, (4, 3)), 5),
             ("fp8", "R6", "eye search, FP8 (R6)", th["eye"], "-", -5), ("bf16", "R5", "eye search, BF16 (R5)", th["eye"], (0, (4, 3)), -9)]
    for lab, nm, label, col, ls, dy in lines:
        rs = sorted([r for r in rows if r["label"] == lab and r["name"] == nm], key=lambda r: int(r["concurrency"]))
        if not rs:
            continue
        xs = [int(r["concurrency"]) for r in rs]
        ys = [float(r["queries_per_s"]) for r in rs]
        ax.plot(xs, ys, color=col, linewidth=2, linestyle=ls, marker="o", markersize=5, zorder=2)
        ax.annotate(label, (xs[-1], ys[-1]), xytext=(10, dy), textcoords="offset points", va="center", fontsize=8.5,
                    color=th["ink"])
    style(ax, th, "concurrent requests", "completed searches / s")
    ax.set_xticks([1, 4, 8, 16])
    title(f, th, "On 640x480 frames, full frame serves more searches per L4",
          "End-to-end searches on the 100-query latency subset; eye search needs ~2 calls per search.")
    return save(f, "throughput", theme)


# ---------------------------------------------------------------- 5. training loss
def loss_fig(F, theme):
    th = THEMES[theme]
    log = list(csv.DictReader(open(R / "step7_train_log.csv")))
    f, ax = fig(th, 7.6, 4.2)
    f.subplots_adjust(left=0.1, right=0.83, top=0.8, bottom=0.14)
    tr = [(int(r["step"]), float(r["loss"])) for r in log if r["loss"]]
    ev = [(int(r["step"]), float(r["eval_loss"])) for r in log if r["eval_loss"]]
    ax.plot(*zip(*tr), color=th["ff"], linewidth=1.5)
    ax.plot(*zip(*ev), color=th["eye"], linewidth=2, marker="o", markersize=5)
    ax.annotate("train loss", tr[-1], xytext=(8, -2), textcoords="offset points", va="center", fontsize=9, color=th["ink"])
    ax.annotate("val loss", ev[-1], xytext=(8, 8), textcoords="offset points", va="center", fontsize=9, color=th["ink"])
    ax.set_ylim(0.4, 1.0)
    style(ax, th, "training step (effective batch 64)", "loss")
    title(f, th, "LoRA fine-tuning: one epoch, 31.5 min on the L4" if F.v["train.minutes"] == "31.5" else
          f"LoRA fine-tuning: one epoch, {F.v['train.minutes']} min on the L4",
          "First logged train loss (step 5) is above the plotted range.")
    return save(f, "loss_curves", theme)


# ---------------------------------------------------------------- 6. example trace: positive2 · food
def trace_fig(F, theme):
    th = THEMES[theme]
    rec = json.loads((R / "demo" / "positive2__food.json").read_text())
    frame = Image.open(REPO_ROOT / rec["image"]).convert("RGB")
    ff, eye = rec["traces"]["full_frame"], rec["traces"]["eye_search"]
    thumb = make_thumbnail(frame, int(rec["settings"]["thumbnail_long_side"]))
    pt_call = eye["calls"][0]
    conf = next(c for c in eye["calls"] if c["role"] == "confirm")
    x0, y0, x1, y1 = conf["region"]
    crop = frame.crop((x0, y0, x1, y1))
    f, axes = plt.subplots(1, 3, figsize=(12, 4.3), dpi=120,
                           gridspec_kw={"width_ratios": [frame.width / frame.height, frame.width / frame.height, 1]})
    f.patch.set_facecolor(th["surface"])
    f.subplots_adjust(left=0.01, right=0.99, top=0.72, bottom=0.03, wspace=0.06)
    sm = rec["summary"]
    # (a) full frame
    a = axes[0]
    a.imshow(frame)
    a.scatter([ff["final_point"][0]], [ff["final_point"][1]], s=160, facecolor="none", edgecolor=th["ff"], linewidth=2.5)
    a.set_title(f"Full frame: {frame.width}x{frame.height}, {sm['full_frame']['vision_tokens']:,} vision tokens\n"
                f"found in {sm['full_frame']['time_median_s']:.2f} s", fontsize=9.5, color=th["ink"], loc="left")
    # (b) the glance
    b = axes[1]
    b.imshow(thumb.resize((frame.width, frame.height), Image.Resampling.NEAREST))
    fx, fy = eye["fixations"][0]
    b.scatter([fx], [fy], s=160, facecolor=th["eye"], edgecolor="white", linewidth=1.5)
    b.text(fx, fy, "1", color="white", ha="center", va="center", fontsize=8, fontweight="bold")
    b.add_patch(matplotlib.patches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=th["eye"], linewidth=1.8))
    b.set_title(f"Eye search, glance: {thumb.width}x{thumb.height} thumbnail ({pt_call['vision_tokens']} tokens)\n"
                f"points at the apple, then crops there", fontsize=9.5, color=th["ink"], loc="left")
    # (c) the fixation crop + final point
    c = axes[2]
    c.imshow(crop)
    px, py = eye["final_point"][0] - x0, eye["final_point"][1] - y0
    c.scatter([px], [py], s=160, facecolor="none", edgecolor=th["eye"], linewidth=2.5)
    c.set_title(f"Fixation 1: sharp {crop.width}x{crop.height} crop\n({conf['vision_tokens']} tokens), confirmed \"{eye['evidence']}\"\n"
                f"found in {sm['eye_search']['time_median_s']:.2f} s, {sm['eye_search']['vision_tokens']} tokens in all",
                fontsize=9.5, color=th["ink"], loc="left")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(th["axis"])
    title(f, th, f"One search, measured: \"{rec['query']}\" on {Path(rec['image']).name}",
          "Median of 5 runs, concurrency 1, fine-tuned FP8 on one L4. Circles: each strategy's final point.")
    return save(f, "trace_positive2_food", theme)


def main() -> int:
    F, _ = compute()
    for theme in THEMES:
        for fn in (hero, scaling_fig, sweep_fig, throughput_fig, loss_fig, trace_fig):
            p = fn(F, theme)
            print(f"{p.relative_to(REPO_ROOT)} ({p.stat().st_size / 1e3:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
