#!/usr/bin/env python
"""Step 5 eye-search sweeps (test split, base BF16): CSV + found-vs-pixels figure.

Thumbnail long side 64/96/128/192 (crop 224) and crop 160/224/288 (thumbnail 96). The default
setting (96 / 224) is R4_eye_search filtered to test. R1-R3 (test) are reference markers.
Writes results/step5_sweeps.csv and results/step5_sweep_found_vs_pixels.png.
"""

from __future__ import annotations

import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from foveated_search.config import REPO_ROOT  # noqa: E402
from foveated_search.eval.metrics import summarize  # noqa: E402
from foveated_search.eval.runner import eval_config  # noqa: E402
from foveated_search.eval.splits import split_ids  # noqa: E402
from foveated_search.eval.table import load_run  # noqa: E402

# (sweep, run, thumbnail long side, crop)
SETTINGS = [
    ("thumbnail", "sweep_thumb64", 64, 224), ("thumbnail", "R4_eye_search", 96, 224),
    ("thumbnail", "sweep_thumb128", 128, 224), ("thumbnail", "sweep_thumb192", 192, 224),
    ("crop", "sweep_crop160", 96, 160), ("crop", "R4_eye_search", 96, 224), ("crop", "sweep_crop288", 96, 288),
]
REFS = [("R1 full frame", "R1_full_frame"), ("R2 sliding window", "R2_sliding_window"), ("R3 downscale", "R3_downscale")]

# Reference palette (dataviz skill), validated: blue/orange pass CVD + normal-vision + contrast (light).
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0", "#fcfcfb"


def main() -> None:
    cfg = eval_config()
    test = split_ids(cfg, "test")
    h = cfg["harness"]

    def summ(run: str) -> dict:
        traces = [t for t in load_run(run)[0] if t["meta"]["query_id"] in test]
        return summarize(traces, int(h["bootstrap_resamples"]), float(h["ci_level"]), int(cfg["seed"]))

    rows = []
    for sweep, run, L, F in SETTINGS:
        s = summ(run)
        rows.append({"sweep": sweep, "run": run, "thumbnail_long_side": L, "crop": F, "n": s["n"],
                     "found": round(s["found"], 4), "found_ci_lo": round(s["found_ci"][0], 4),
                     "found_ci_hi": round(s["found_ci"][1], 4), "fp_rate": round(s["fp_rate"], 4),
                     "accuracy": round(s["accuracy"], 4), "wrong_location": round(s["wrong_location"], 4),
                     "pixels_frac": round(s["pixels_frac"], 4), "pixels_unique_frac": round(s["pixels_unique_frac"], 4),
                     "vision_tokens": round(s["vision_tokens"], 1), "output_tokens": round(s["completion_tokens"], 1),
                     "calls": round(s["calls"], 3), "fixations": round(s["fixations"], 3)})
    refs = []
    for label, run in REFS:
        s = summ(run)
        refs.append({"sweep": "reference", "run": run, "thumbnail_long_side": "", "crop": "", "n": s["n"],
                     "found": round(s["found"], 4), "found_ci_lo": round(s["found_ci"][0], 4),
                     "found_ci_hi": round(s["found_ci"][1], 4), "fp_rate": round(s["fp_rate"], 4),
                     "accuracy": round(s["accuracy"], 4), "wrong_location": round(s["wrong_location"], 4),
                     "pixels_frac": round(s["pixels_frac"], 4), "pixels_unique_frac": round(s["pixels_unique_frac"], 4),
                     "vision_tokens": round(s["vision_tokens"], 1), "output_tokens": round(s["completion_tokens"], 1),
                     "calls": round(s["calls"], 3), "fixations": round(s["fixations"], 3), "label": label})
    out_csv = REPO_ROOT / "results/step5_sweeps.csv"
    keys = list(rows[0]) + ["label"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows + refs)

    # ------------------------------------------------------------------ figure
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5})
    fig, ax = plt.subplots(figsize=(8.4, 5.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for sweep, color, lab_key, unit in [("thumbnail", BLUE, "thumbnail_long_side", "px thumb"), ("crop", ORANGE, "crop", "px crop")]:
        pts = [r for r in rows if r["sweep"] == sweep]
        x = [100 * r["pixels_frac"] for r in pts]
        y = [100 * r["found"] for r in pts]
        lo = [100 * (r["found"] - r["found_ci_lo"]) for r in pts]
        hi = [100 * (r["found_ci_hi"] - r["found"]) for r in pts]
        ax.errorbar(x, y, yerr=[lo, hi], fmt="none", ecolor=color, elinewidth=1, alpha=0.45, capsize=0, zorder=2)
        ax.plot(x, y, "-", color=color, lw=2, zorder=3,
                label="eye search: thumbnail size (crop 224)" if sweep == "thumbnail" else "eye search: crop size (thumbnail 96)")
        ax.scatter(x, y, s=46, color=color, edgecolor=SURFACE, linewidth=2, zorder=4)
        for r, xi, yi in zip(pts, x, y):
            if r["run"] == "R4_eye_search":
                continue  # default point: labelled "R4 default" below
            dy = -13 if sweep == "crop" else 8
            ax.annotate(f"{r[lab_key]}", (xi, yi), textcoords="offset points", xytext=(6, dy), color=INK2, fontsize=9)
    r4 = next(r for r in rows if r["run"] == "R4_eye_search")
    ax.annotate("R4 default\n(96 / 224)", (100 * r4["pixels_frac"], 100 * r4["found"]), textcoords="offset points",
                xytext=(-62, 10), color=INK, fontsize=9, fontweight="bold")
    for r in refs:
        x, y = 100 * r["pixels_frac"], 100 * r["found"]
        ax.errorbar([x], [y], yerr=[[100 * (r["found"] - r["found_ci_lo"])], [100 * (r["found_ci_hi"] - r["found"])]],
                    fmt="none", ecolor=MUTED, elinewidth=1, alpha=0.6, zorder=2)
        ax.scatter([x], [y], s=60, facecolor=SURFACE, edgecolor=INK2, linewidth=1.8, marker="D", zorder=4)
        ax.annotate(f"{r['label']}\nFP {100 * r['fp_rate']:.0f}% on negatives", (x, y), textcoords="offset points",
                    xytext=(-10, -30) if x > 90 else (-12, 4), ha="right", color=INK2, fontsize=8.5)
    ax.scatter([], [], s=60, facecolor=SURFACE, edgecolor=INK2, linewidth=1.8, marker="D", label="baselines R1-R3")
    ax.set_xlabel("mean pixels sampled per query (% of frame)", color=INK2)
    ax.set_ylabel("found % on positives (95% CI)", color=INK2)
    ax.set_title("Eye search on COCO (test, base Qwen3-VL-2B, BF16): found vs pixels", color=INK, fontsize=12,
                 loc="left", pad=12)
    ax.set_xlim(0, 110)
    ax.set_ylim(40, 100)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    for sp in ["left", "bottom"]:
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK2)
    leg = ax.legend(loc="lower right", frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color(INK2)
    fps = [100 * r["fp_rate"] for r in rows]
    fig.text(0.01, 0.01, f"Eye search FP on negatives: {min(fps):.1f}-{max(fps):.1f}% at every setting (n = 90 negatives; "
             "360 positives).\nPixels count every image sent (thumbnail re-sent per fixation; R2 tiles overlap, "
             "so > 100%).", color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    out_png = REPO_ROOT / "results/step5_sweep_found_vs_pixels.png"
    fig.savefig(out_png, facecolor=SURFACE)
    print(f"wrote {out_csv.relative_to(REPO_ROOT)}, {out_png.relative_to(REPO_ROOT)}")
    for r in rows + refs:
        print({k: r[k] for k in ["run", "thumbnail_long_side", "crop", "found", "fp_rate", "accuracy",
                                 "wrong_location", "pixels_frac", "vision_tokens", "calls", "fixations"]})


if __name__ == "__main__":
    main()
