#!/usr/bin/env python
"""Step 8b figure: results/step8b_scaling.csv -> results/step8b_scaling.png (three panels vs frame size)."""

from __future__ import annotations

import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from foveated_search.config import REPO_ROOT  # noqa: E402

INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0", "#fcfcfb"
# validated reference palette; 3 slots are below 3:1 contrast and one adjacent pair is in the CVD floor band,
# so every line also gets its own marker shape and a direct end label (+ the CSV / table.md as table view)
STYLE = {"full frame": ("#2a78d6", "o", "full frame"),
         "downscale (eye-96 budget)": ("#eb6834", "s", "downscale"),
         "eye search 96 px": ("#1baf7a", "^", "eye 96 px"),
         "eye search 192 px": ("#eda100", "D", "eye 192 px"),
         "eye search 288 px": ("#e87ba4", "v", "eye 288 px")}


def main() -> None:
    rows = list(csv.DictReader(open(REPO_ROOT / "results/step8b_scaling.csv")))
    sizes = list(dict.fromkeys(r["frame"] for r in rows))
    x = list(range(len(sizes)))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.9), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    panels = [("accuracy", "accuracy % (positives found + negatives rejected)", lambda v: 100 * float(v), False),
              ("vision_tokens", "vision tokens per search (log scale)", float, True),
              ("searches_per_s_c16", "completed searches / s at concurrency 16", float, False)]
    for ax, (key, title, conv, log) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        ends = []
        for strat, (color, marker, short) in STYLE.items():
            pts = [(sizes.index(r["frame"]), conv(r[key])) for r in rows if r["strategy"] == strat and r[key] not in ("", None)]
            if not pts:
                continue
            ax.plot([p for p, _ in pts], [v for _, v in pts], "-", color=color, lw=2, marker=marker, ms=7, mec=SURFACE,
                    mew=1.5, label=strat)
            ends.append((pts[-1][1], short, color))
        if key == "accuracy":  # direct end labels, nudged apart
            placed = []
            for v, short, color in sorted(ends, key=lambda e: -e[0]):
                y = v
                while any(abs(y - p) < 3.2 for p in placed):
                    y -= 0.5
                placed.append(y)
                ax.annotate(short, (x[-1], v), xytext=(x[-1] + 0.08, y), textcoords="data", va="center",
                            color=INK2, fontsize=8.5)
            ax.set_xlim(-0.2, len(sizes) - 1 + 0.55)
        else:
            ax.set_xlim(-0.2, len(sizes) - 0.8)
        if log:
            ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace(" (", "\n(") for s in sizes], color=INK2)
        ax.set_title(title, loc="left", color=INK, fontsize=10.5)
        ax.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        for sp in ["left", "bottom"]:
            ax.spines[sp].set_color(GRID)
        ax.tick_params(colors=INK2)
    leg = axes[2].legend(loc="lower left", frameon=False, fontsize=8.5)
    for t in leg.get_texts():
        t.set_color(INK2)
    fig.suptitle("How search strategies scale with frame size (fine-tuned Qwen3-VL-2B, FP8, one L4; same 308 COCO "
                 "test queries as 1x1 frames and as 2x2 / 3x3 mosaics)", x=0.01, ha="left", color=INK, fontsize=11.5)
    fig.text(0.01, 0.01, "Full frame at native resolution (up to 300 / 1,200 / 2,700 vision tokens). Downscale budget = "
             "eye-96's mean pixels at each size. Data: results/step8b_scaling.csv.", color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(REPO_ROOT / "results/step8b_scaling.png", facecolor=SURFACE)
    print("wrote results/step8b_scaling.png")


if __name__ == "__main__":
    main()
