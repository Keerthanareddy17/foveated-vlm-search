"""Step 7: push the merged model to a PRIVATE Hugging Face repo.

  python -m foveated_search.train.push_hub     (make push-hub)

Checks the logged-in token's role (write access) WITHOUT printing it, refuses to push to an existing
public repo, never deletes anything on the Hub. Writes a model card (README.md) into the merged folder.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from foveated_search.config import REPO_ROOT
from foveated_search.train.train_lora import train_config


def can_write(info: dict, namespace: str) -> tuple[bool, str]:
    tok = (info.get("auth") or {}).get("accessToken") or {}
    role = tok.get("role")
    if role == "write":
        return True, "write"
    if role == "fineGrained":
        scoped = (tok.get("fineGrained") or {}).get("scoped") or []
        for s in scoped:
            ent = s.get("entity") or {}
            if ent.get("name") == namespace and "repo.write" in (s.get("permissions") or []):
                return True, "fineGrained (repo.write on user namespace)"
        return False, "fineGrained without repo.write on the user namespace"
    return False, str(role)


def model_card(cfg, repo_id: str, meta: dict) -> str:
    t = cfg["train"]
    td = meta.get("train", {})
    return f"""---
license: apache-2.0
base_model: {t['base_model']}
library_name: transformers
pipeline_tag: image-text-to-text
tags:
- qwen3-vl
- visual-search
- grounding
- lora-merged
datasets:
- coco
---

# {repo_id.split('/')[-1]}

{t['base_model']} fine-tuned (LoRA, merged into BF16 weights) for **low-pixel visual search**: finding a
queried object in a camera frame while looking at as few pixels as possible. Part of the foveated VLM search
project: {t['github_url']}

## What it was trained for

Three skills, each matching one call of the project's "eye search" loop:

1. **Where to look**: from a small thumbnail of the whole frame (long side 64-192 px), point at the queried
   object, or report that it is absent.
2. **Look again**: the same thumbnail plus a list of already-checked locations that did not contain the object;
   point at a different location, or report absent.
3. **Confirm**: on a sharp crop (160-288 px) of the original frame, answer whether the object is there and, if
   so, point at it inside the crop.

Queries are COCO category names (80 classes) or "food". Points use Qwen3-VL's relative 0-1000 coordinates on the
image as sent, in JSON: `{{"point_2d": [x, y]}}` / `{{"absent": true}}` and `{{"y": false}}` /
`{{"y": true, "point_2d": [x, y]}}`. The exact prompts are in the project repository (`configs/prompts.yaml`).

## Training data

Auto-generated from **COCO train2017** annotations (no overlap with the project's COCO val2017 evaluation images):
{td.get('train_samples', 'n/a')} training samples ({td.get('val_samples', 'n/a')} validation) built from
~7.5k images. On-object target points come from the instance segmentation masks; negatives exclude the
query's whole supercategory, and "food" negatives also exclude food-adjacent scenes (tables, bowls, cups, ...)
because COCO does not label every kind of food.

## Training

LoRA r={t['lora']['r']}, alpha={t['lora']['alpha']}, dropout={t['lora']['dropout']} on all linear projections of the
language-model decoder layers; vision tower and vision-language merger frozen. AdamW, learning rate
{t['learning_rate']}, cosine schedule, {t['epochs']} epoch, effective batch
{int(t['per_device_batch_size']) * int(t['grad_accum'])}, BF16, one NVIDIA L4.
Final validation loss: {td.get('final_eval_loss', 'n/a')}.

## Use

Same usage as the base model (transformers or vLLM). For the token counts the project relies on, set the image
processor's minimum pixels to 1024 (`size = {{"shortest_edge": 1024, "longest_edge": 1228800}}`), so a 96x64
thumbnail costs 6 vision tokens instead of being upscaled.

## Limitations

Trained and evaluated on COCO images only; "food" is limited to COCO's 10 food classes, so real but unlabelled
food can be scored as a miss. Evaluation results: see the project repository.
"""


def main() -> None:
    from huggingface_hub import HfApi

    cfg = train_config()
    t = cfg["train"]
    folder = Path(cfg.get_key("paths.models_dir")) / t["merged_name"]
    if not (folder / "config.json").exists():
        sys.exit(f"merged model not found at {folder}: run make merge first")
    api = HfApi()
    info = api.whoami()
    user = info["name"]
    ok, why = can_write(info, user)
    print(f"[hub] logged in as {user}; token role: {why} (token not shown)")
    if not ok:
        sys.exit("[hub] STOP: the token has no write access to your namespace. Create a write token and log in again.")
    repo_id = f"{user}/{t['hub_repo_name']}"
    if api.repo_exists(repo_id):
        if not api.model_info(repo_id).private:
            sys.exit(f"[hub] STOP: {repo_id} already exists and is PUBLIC; not pushing.")
        print(f"[hub] {repo_id} exists (private): uploading a new revision, nothing is deleted")
    else:
        api.create_repo(repo_id, private=True, repo_type="model")
        print(f"[hub] created private repo {repo_id}")
    train_meta = json.loads((Path(cfg.get_key("paths.checkpoints_dir")) / t["out_subdir"] / t["run_name"] /
                             "adapter" / "train_meta.json").read_text())
    log = REPO_ROOT / "results/step7_train_log.csv"
    final_eval = None
    if log.exists():
        import csv

        evals = [r for r in csv.DictReader(open(log)) if r["eval_loss"]]
        final_eval = round(float(evals[-1]["eval_loss"]), 4) if evals else None
    meta = {"train": {"train_samples": train_meta["train_samples"], "val_samples": train_meta["val_samples"],
                      "final_eval_loss": final_eval}}
    (folder / "README.md").write_text(model_card(cfg, repo_id, meta))
    commit = api.upload_folder(folder_path=str(folder), repo_id=repo_id, repo_type="model",
                               commit_message="Upload merged Qwen3-VL-2B foveated-search model",
                               allow_patterns=["*.json", "*.safetensors", "*.txt", "README.md"],
                               ignore_patterns=["merge_meta.json"])
    mi = api.model_info(repo_id, files_metadata=True)
    files = {s.rfilename: s.size for s in mi.siblings}
    res = {"repo_id": repo_id, "private": mi.private, "commit": getattr(commit, "oid", None), "files": files,
           "total_gb": round(sum(v or 0 for v in files.values()) / 1e9, 3)}
    (REPO_ROOT / "results/step7_hub_push.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
