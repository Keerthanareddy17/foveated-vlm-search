"""Step 6 check in .venv-train: samples go through the Qwen3-VL processor + chat template cleanly,
vision tokens match the server settings (96 px thumbnail = 6 tokens), only the assistant span is
trained, and token counts per type (for Step 7's time estimate).

  .venv-train/bin/python -m foveated_search.train.check_data [--per-type 400]
Writes results/step6_token_stats.json.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from foveated_search.config import REPO_ROOT, load_config
from foveated_search.train.collate import IGNORE, encode, load_processor


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-type", type=int, default=400)
    a = ap.parse_args()
    cfg = load_config(overlays=["configs/search.yaml", "configs/prompts.yaml", "configs/eval.yaml",
                                "configs/train_data.yaml"])
    root = Path(cfg.get_key("paths.data_dir")) / cfg.get_key("train_data.out_subdir")
    samples = [json.loads(line) for line in open(root / "train.jsonl")]
    proc = load_processor(cfg)
    merge = int(proc.image_processor.merge_size)
    rng = random.Random(int(cfg["seed"]))
    by_type = defaultdict(list)
    for s in samples:
        by_type[s["type"]].append(s)

    stats, checks = {}, {"thumb96_tokens": [], "vision_mismatch": 0, "label_mismatch": 0, "n": 0}
    shown = None
    for t, ss in by_type.items():
        tot, vis, ans = [], [], []
        for s in rng.sample(ss, min(a.per_type, len(ss))):
            e = encode(s, root, proc)
            g = e["image_grid_thw"][0].tolist()
            v = g[0] * g[1] * g[2] // merge**2
            m = s["meta"]
            # the processor must not resize: grid * 16 == the size we sent (sides are multiples of 32)
            if m["sent_w"] % 32 == 0 and m["sent_h"] % 32 == 0 and (g[2] * 16, g[1] * 16) != (m["sent_w"], m["sent_h"]):
                checks["vision_mismatch"] += 1
            if t != "confirm" and m["thumb_long_side"] == 96 and m["sent_w"] * m["sent_h"] == 96 * 64:
                checks["thumb96_tokens"].append(v)
            trained = e["labels"][e["labels"] != IGNORE]
            decoded = proc.tokenizer.decode(trained)
            target = s["messages"][1]["content"][0]["text"]
            if decoded != target + "<|im_end|>":
                checks["label_mismatch"] += 1
            tot.append(len(e["input_ids"]))
            vis.append(v)
            ans.append(len(trained))
            checks["n"] += 1
            if shown is None and t == "look_again":
                ids = e["input_ids"]
                k = e["n_prompt"]
                shown = {"id": s["id"], "type": t,
                         "masked (not trained)": proc.tokenizer.decode(ids[:k]).replace("<|image_pad|>" * v, f"<|image_pad|>x{v}"),
                         "trained": proc.tokenizer.decode(ids[k:][e["labels"][k:] != IGNORE]),
                         "masked tail": proc.tokenizer.decode(ids[k:][e["labels"][k:] == IGNORE])}
        stats[t] = {"n": len(tot), "mean": float(np.mean(tot)), "p95": float(np.percentile(tot, 95)),
                    "vision_mean": float(np.mean(vis)), "assistant_mean": float(np.mean(ans)),
                    "total_tokens_full_dataset_est": float(np.mean(tot)) * len(ss), "samples_in_train": len(ss)}
    out = {"processor_size": dict(proc.image_processor.size), "per_type": stats,
           "thumb96_vision_tokens": sorted(set(checks["thumb96_tokens"])),
           "thumb96_checked": len(checks["thumb96_tokens"]), "samples_checked": checks["n"],
           "processor_resized_grid_aligned_images": checks["vision_mismatch"],
           "label_span_mismatches": checks["label_mismatch"], "example": shown,
           "train_tokens_per_epoch_est": sum(v["total_tokens_full_dataset_est"] for v in stats.values())}
    (REPO_ROOT / "results/step6_token_stats.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "example"}, indent=1))
    print("\n=== masked / trained spans for one sample ===")
    for k, v in shown.items():
        print(f"--- {k}:\n{v}")
    ok = out["thumb96_vision_tokens"] == [6] and not checks["vision_mismatch"] and not checks["label_mismatch"]
    print("\nCHECK PASSED" if ok else "\nCHECK FAILED")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
