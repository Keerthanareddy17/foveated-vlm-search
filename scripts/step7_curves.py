#!/usr/bin/env python
"""Step 7: training log -> results/step7_train_log.csv + results/step7_loss_curves.png."""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from foveated_search.config import REPO_ROOT, load_config  # noqa: E402

BLUE, ORANGE = "#2a78d6", "#eb6834"          # validated pair (dataviz reference palette, light)
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0", "#fcfcfb"


def main() -> None:
    cfg = load_config(overlays=["configs/train.yaml"])
    t = cfg["train"]
    src = Path(cfg.get_key("paths.checkpoints_dir")) / t["out_subdir"] / t["run_name"] / "train_log.csv"
    dst = REPO_ROOT / "results/step7_train_log.csv"
    shutil.copyfile(src, dst)
    rows = list(csv.DictReader(open(dst)))
    tr = [(int(r["step"]), float(r["loss"])) for r in rows if r["loss"]]
    ev = [(int(r["step"]), float(r["eval_loss"])) for r in rows if r["eval_loss"]]

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5})
    fig, ax = plt.subplots(figsize=(8.4, 4.8), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.plot([s for s, _ in tr], [v for _, v in tr], color=BLUE, lw=1.6, label="train loss (every 5 steps)")
    ax.plot([s for s, _ in ev], [v for _, v in ev], color=ORANGE, lw=2, marker="o", ms=6, mec=SURFACE, mew=2,
            label=f"val loss (full val split, {len(ev)} evaluations)")
    if ev:
        s, v = ev[-1]
        ax.annotate(f"final val {v:.3f}", (s, v), textcoords="offset points", xytext=(-8, 12), ha="right",
                    color=INK2, fontsize=9)
    steps = max(s for s, _ in tr + ev)
    ax.set_xlabel(f"optimizer step (effective batch {int(t['per_device_batch_size']) * int(t['grad_accum'])}; "
                  f"{steps} steps = {t['epochs']} epoch)", color=INK2)
    ax.set_ylabel("loss (answer tokens only)", color=INK2)
    ax.set_title(f"LoRA fine-tuning of Qwen3-VL-2B for low-pixel search (r={t['lora']['r']}, lr={t['learning_rate']})",
                 loc="left", color=INK, fontsize=12, pad=12)
    lo = min(v for _, v in tr + ev)
    ax.set_ylim(max(0, lo - 0.05), min(max(v for _, v in tr), lo + 0.6))
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    for sp in ["left", "bottom"]:
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK2)
    leg = ax.legend(frameon=False, loc="upper right", fontsize=9)
    for x in leg.get_texts():
        x.set_color(INK2)
    first = tr[0][1]
    fig.text(0.01, 0.01, f"First logged train loss {first:.2f} (step {tr[0][0]}) is above the plot range. "
             "Data: results/step7_train_log.csv.", color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out = REPO_ROOT / "results/step7_loss_curves.png"
    fig.savefig(out, facecolor=SURFACE)
    print(f"wrote {dst.relative_to(REPO_ROOT)}, {out.relative_to(REPO_ROOT)}; train points {len(tr)}, val points {len(ev)}")


if __name__ == "__main__":
    main()
