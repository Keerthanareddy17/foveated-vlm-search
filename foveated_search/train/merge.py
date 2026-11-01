"""Step 7: merge the LoRA adapter into the BF16 base and save the merged model on the local disk.

  python -m foveated_search.train.merge        (make merge; CPU, after training)

Processor / tokenizer / chat-template / preprocessor files are copied byte-for-byte from the base
snapshot (same min/max pixels as the base; our server still passes its own --mm-processor-kwargs).
Prints a diff of config.json and generation_config.json against the base, and checks that every
vision-tower and merger tensor is bit-identical to the base (LoRA touched only the decoder).
Writes <merged>/merge_meta.json and results/step7_merge_diff.json.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import torch

from foveated_search.config import REPO_ROOT
from foveated_search.train.train_lora import train_config

COPY_FROM_BASE = ["preprocessor_config.json", "video_preprocessor_config.json", "chat_template.json",
                  "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "generation_config.json"]


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def json_diff(a, b, path="") -> list[str]:
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            p = f"{path}.{k}" if path else k
            if k not in a:
                out.append(f"+ {p} = {json.dumps(b[k])[:120]}")
            elif k not in b:
                out.append(f"- {p} = {json.dumps(a[k])[:120]}")
            else:
                out += json_diff(a[k], b[k], p)
        return out
    return [] if a == b else [f"~ {path}: {json.dumps(a)[:120]} -> {json.dumps(b)[:120]}"]


def main() -> None:
    from huggingface_hub import snapshot_download
    from peft import PeftModel
    from safetensors import safe_open
    from transformers import Qwen3VLForConditionalGeneration

    cfg = train_config()
    t = cfg["train"]
    adapter = Path(cfg.get_key("paths.checkpoints_dir")) / t["out_subdir"] / t["run_name"] / "adapter"
    out = Path(cfg.get_key("paths.models_dir")) / t["merged_name"]
    base_dir = Path(snapshot_download(t["base_model"]))
    print(f"[merge] base {base_dir}\n[merge] adapter {adapter}\n[merge] out {out}", flush=True)

    base = Qwen3VLForConditionalGeneration.from_pretrained(t["base_model"], dtype=torch.bfloat16, device_map="cpu")
    merged = PeftModel.from_pretrained(base, str(adapter)).merge_and_unload()
    if out.exists():
        shutil.rmtree(out)  # our own previous local merge output only
    merged.save_pretrained(str(out), safe_serialization=True)
    for f in COPY_FROM_BASE:
        if (base_dir / f).exists():
            shutil.copyfile(base_dir / f, out / f)

    # ---- diffs against the base. transformers 5.x re-serializes config.json (rope_scaling -> rope_parameters,
    # dtype fields, version); the architecture and tensor keys are identical, so the base config.json is shipped
    # byte-for-byte (zero diff, readable by transformers 4.57+ and vLLM). The re-serialization diff is recorded.
    diff = {}
    for f in ["config.json", "generation_config.json"]:
        diff[f] = json_diff(json.loads((base_dir / f).read_text()), json.loads((out / f).read_text()))
    shutil.copyfile(base_dir / "config.json", out / "config.json")
    final_config_identical = _sha(base_dir / "config.json") == _sha(out / "config.json")
    copied = {f: _sha(base_dir / f) == _sha(out / f) for f in COPY_FROM_BASE if (base_dir / f).exists()}
    base_files = sorted(p.name for p in base_dir.iterdir() if p.is_file() and not p.name.startswith("."))
    out_files = sorted(p.name for p in out.iterdir() if p.is_file())

    # ---- vision tower untouched, decoder changed
    def tensors(d: Path) -> dict[str, Path]:
        m = {}
        for st in d.glob("*.safetensors"):
            with safe_open(str(st), "pt") as f:
                for k in f.keys():
                    m[k] = st
        return m

    tb, to = tensors(base_dir), tensors(out)
    vis_same = dec_changed = 0
    vis_keys = [k for k in tb if ".visual." in k]
    dec_keys = [k for k in tb if ".language_model.layers." in k and k.endswith("proj.weight")]
    with_handles = {}

    def get(path: Path, key: str):
        if path not in with_handles:
            with_handles[path] = safe_open(str(path), "pt")
        return with_handles[path].get_tensor(key)

    for k in vis_keys:
        vis_same += bool(torch.equal(get(tb[k], k), get(to[k], k)))
    for k in dec_keys:
        dec_changed += not bool(torch.equal(get(tb[k], k), get(to[k], k)))
    keys_equal = set(tb) == set(to)

    res = {"config_diff_as_reserialized_by_transformers_5": diff, "shipped_config_identical_to_base": final_config_identical,
           "copied_files_identical": copied, "base_files": base_files, "merged_files": out_files,
           "tensor_keys_identical": keys_equal, "vision_tensors": len(vis_keys), "vision_tensors_identical": vis_same,
           "decoder_proj_weights": len(dec_keys), "decoder_proj_weights_changed": dec_changed,
           "merged_size_gb": round(sum(p.stat().st_size for p in out.glob("*.safetensors")) / 1e9, 3)}
    (out / "merge_meta.json").write_text(json.dumps({**res, "adapter": str(adapter), "base": t["base_model"]}, indent=1) + "\n")
    (REPO_ROOT / "results/step7_merge_diff.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))
    ok = keys_equal and vis_same == len(vis_keys) and dec_changed == len(dec_keys) and all(copied.values()) \
        and final_config_identical
    print("MERGE CHECK PASSED" if ok else "MERGE CHECK FAILED")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
