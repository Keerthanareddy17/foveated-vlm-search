"""Step 7: LoRA fine-tuning of Qwen3-VL-2B on the Step 6 data (.venv-train). Never run with vLLM up.

  python -m foveated_search.train.train_lora --bench 40     # throughput + memory benchmark, then cleaned up
  python -m foveated_search.train.train_lora                # full run (resumes from the latest checkpoint)

Inputs are built ONLY by foveated_search.train.collate.encode (same processor settings as the server,
loss on the assistant answer only); the HF Trainer gets our padding collator and does no image
processing of its own (remove_unused_columns=False). A startup check compares the Trainer's first
batch with our collator applied directly to the same sample indices.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import time
from pathlib import Path

import torch
from torch.utils.data import Dataset

from foveated_search.config import REPO_ROOT, load_config
from foveated_search.train.collate import IGNORE, encode, load_processor

OVERLAYS = ["configs/search.yaml", "configs/prompts.yaml", "configs/eval.yaml", "configs/train_data.yaml",
            "configs/train.yaml"]


def train_config():
    return load_config(overlays=OVERLAYS)


class SampleDataset(Dataset):
    def __init__(self, path: Path, root: Path, proc):
        self.samples = [json.loads(line) for line in open(path)]
        self.root, self.proc = root, proc

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> dict:
        e = encode(self.samples[i], self.root, self.proc)
        e.pop("n_prompt")
        return e


class Collator:
    """Right-pad text; concatenate image patches; stack grids. Nothing else."""

    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch: list[dict]) -> dict:
        n = max(len(b["input_ids"]) for b in batch)

        def pad(key, value):
            out = torch.full((len(batch), n), value, dtype=batch[0][key].dtype)
            for i, b in enumerate(batch):
                out[i, : len(b[key])] = b[key]
            return out

        out = {"input_ids": pad("input_ids", self.pad_id), "attention_mask": pad("attention_mask", 0),
               "labels": pad("labels", IGNORE),
               "pixel_values": torch.cat([b["pixel_values"] for b in batch], 0),
               "image_grid_thw": torch.cat([b["image_grid_thw"] for b in batch], 0)}
        if "mm_token_type_ids" in batch[0]:
            out["mm_token_type_ids"] = pad("mm_token_type_ids", 0)
        return out


def git_commit() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", type=int, default=0, help="run N optimizer steps, report speed/memory, clean up")
    a = ap.parse_args()

    from peft import LoraConfig, get_peft_model
    from transformers import Qwen3VLForConditionalGeneration, Trainer, TrainerCallback, TrainingArguments, set_seed

    cfg = train_config()
    t = cfg["train"]
    seed = int(cfg["seed"])
    set_seed(seed)
    data_root = Path(cfg.get_key("paths.data_dir")) / cfg.get_key("train_data.out_subdir")
    run_dir = Path(cfg.get_key("paths.checkpoints_dir")) / t["out_subdir"] / (f"bench_{t['run_name']}" if a.bench else t["run_name"])
    drive_dir = Path(cfg.get_key("paths.drive_adapters_dir")) / t["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)

    proc = load_processor(cfg)
    train_ds = SampleDataset(data_root / "train.jsonl", data_root, proc)
    val_ds = SampleDataset(data_root / "val.jsonl", data_root, proc)
    collator = Collator(proc.tokenizer.pad_token_id)

    model = Qwen3VLForConditionalGeneration.from_pretrained(t["base_model"], dtype=getattr(torch, t["dtype"]),
                                                            attn_implementation=t["attn_implementation"])
    lcfg = LoraConfig(r=int(t["lora"]["r"]), lora_alpha=int(t["lora"]["alpha"]), lora_dropout=float(t["lora"]["dropout"]),
                      target_modules=t["lora"]["target_modules_regex"], bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lcfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    lora_mods = [n for n, _ in model.named_modules() if n.endswith("lora_A")]
    assert lora_mods and all(".language_model.layers." in n for n in lora_mods), "LoRA outside the decoder layers"
    assert not any(p.requires_grad for n, p in model.named_parameters() if ".visual." in n), "vision tower not frozen"
    print(f"[train] trainable params {trainable:,} / {total:,} ({100 * trainable / total:.3f}%), "
          f"{len(lora_mods)} LoRA modules (decoder only; vision tower + mergers frozen)", flush=True)
    if t["gradient_checkpointing"]:
        model.enable_input_require_grads()

    eff = int(t["per_device_batch_size"]) * int(t["grad_accum"])
    steps_per_epoch = math.ceil(len(train_ds) / eff)
    total_steps = a.bench or steps_per_epoch * int(t["epochs"])
    eval_steps = max(1, round(steps_per_epoch * float(t["eval_every_frac"])))
    save_steps = max(1, round(steps_per_epoch * float(t["save_every_frac"])))
    args = TrainingArguments(
        output_dir=str(run_dir), seed=seed, data_seed=seed, bf16=True,
        per_device_train_batch_size=int(t["per_device_batch_size"]), gradient_accumulation_steps=int(t["grad_accum"]),
        per_device_eval_batch_size=int(t["eval_batch_size"]),
        learning_rate=float(t["learning_rate"]), lr_scheduler_type=t["lr_scheduler"], warmup_steps=float(t["warmup_ratio"]),
        weight_decay=float(t["weight_decay"]), max_grad_norm=float(t["max_grad_norm"]), optim="adamw_torch",
        num_train_epochs=int(t["epochs"]), max_steps=a.bench if a.bench else -1,
        logging_steps=int(t["log_every_steps"]), eval_strategy="no" if a.bench else "steps", eval_steps=eval_steps,
        save_strategy="steps", save_steps=a.bench or save_steps, save_total_limit=int(t["save_total_limit"]),
        gradient_checkpointing=bool(t["gradient_checkpointing"]),
        dataloader_num_workers=int(t["dataloader_workers"]), dataloader_persistent_workers=True,
        remove_unused_columns=False, report_to=[], disable_tqdm=True, label_names=["labels"],
    )

    log_csv = run_dir / "train_log.csv"

    class LogCallback(TrainerCallback):
        """CSV log on the local disk (copied to Drive on every eval and at the end)."""
        t0 = time.perf_counter()

        def on_log(self, args_, state, control, logs=None, **kw):
            if not logs:
                return
            row = {"step": state.global_step, "epoch": round(state.epoch or 0, 4), "time_s": round(time.perf_counter() - self.t0, 1),
                   "loss": logs.get("loss"), "eval_loss": logs.get("eval_loss"), "lr": logs.get("learning_rate"),
                   "grad_norm": logs.get("grad_norm"), "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
            new = not log_csv.exists()
            with open(log_csv, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(row))
                if new:
                    w.writeheader()
                w.writerow(row)
            loss = logs.get("loss", logs.get("eval_loss"))
            if loss is not None and not math.isfinite(loss):
                print(f"[train] NON-FINITE LOSS at step {state.global_step}: {loss}", flush=True)
                control.should_training_stop = True
            print(f"[train] {json.dumps(row)}", flush=True)

        def on_evaluate(self, args_, state, control, **kw):
            if not a.bench and drive_dir.parent.exists():
                drive_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy(log_csv, drive_dir / "train_log.csv")

    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=None if a.bench else val_ds,
                      data_collator=collator, callbacks=[LogCallback()])

    # --- the Trainer's dataloader pipeline (same collator, accelerate wrapping) yields exactly our
    # collator's output: checked on an unshuffled slice of the training set, so indices are known
    n_chk = int(t["eval_batch_size"])
    probe = torch.utils.data.Subset(train_ds, list(range(n_chk)))
    got = next(iter(trainer.get_eval_dataloader(probe)))
    ref = collator([train_ds[i] for i in range(n_chk)])
    same = set(got) == set(ref) and all(torch.equal(got[k].cpu(), ref[k]) for k in ref)
    print(f"[train] Trainer dataloader batch == collate.encode + Collator (first {n_chk} train samples): {same}", flush=True)
    assert same, "Trainer batch differs from our collator output"

    resume = None if a.bench else (max(run_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]), default=None))
    if resume:
        print(f"[train] resuming from {resume}", flush=True)
    print(f"[train] {len(train_ds)} train / {len(val_ds)} val samples; effective batch {eff}; "
          f"{steps_per_epoch} steps/epoch; total {total_steps}; eval every {eval_steps}, save every {save_steps}", flush=True)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    trainer.train(resume_from_checkpoint=str(resume) if resume else None)
    dt = time.perf_counter() - t0
    # (transformers 5.17's Trainer also evaluates at the final step: the last val point is the final adapter)
    peak = torch.cuda.max_memory_allocated() / 1e9
    reserved = torch.cuda.max_memory_reserved() / 1e9

    if a.bench:
        samples = a.bench * eff
        # token throughput from the real token counts of the samples consumed (Step 6 stats)
        tok = json.loads((REPO_ROOT / "results/step6_token_stats.json").read_text())
        mean_tok = tok["train_tokens_per_epoch_est"] / sum(v["samples_in_train"] for v in tok["per_type"].values())
        tokens = samples * mean_tok
        sps = samples / dt
        epoch_s = len(train_ds) / sps
        evals = 1 / float(t["eval_every_frac"])
        res = {"bench_steps": a.bench, "seconds": round(dt, 1), "samples_per_s": round(sps, 2),
               "tokens_per_s_est": round(tokens / dt), "peak_mem_allocated_gb": round(peak, 2),
               "peak_mem_reserved_gb": round(reserved, 2), "per_device_batch": int(t["per_device_batch_size"]),
               "grad_accum": int(t["grad_accum"]), "gradient_checkpointing": bool(t["gradient_checkpointing"]),
               "epoch_train_min_est": round(epoch_s / 60, 1), "n_evals": evals, "trainable_params": trainable}
        print("[bench] " + json.dumps(res), flush=True)
        (REPO_ROOT / "results/step7_bench.json").write_text(json.dumps(res, indent=1) + "\n")
        shutil.rmtree(run_dir)  # benchmark checkpoints are not kept
        print(f"[bench] removed {run_dir}", flush=True)
        return

    final = run_dir / "adapter"
    trainer.save_model(str(final))
    proc.save_pretrained(str(final / "processor"))
    meta = {"run_name": t["run_name"], "seconds": round(dt, 1), "peak_mem_allocated_gb": round(peak, 2),
            "peak_mem_reserved_gb": round(reserved, 2), "trainable_params": trainable, "total_params": total,
            "steps": trainer.state.global_step, "train_samples": len(train_ds), "val_samples": len(val_ds),
            "git_commit": git_commit(), "config": cfg["train"], "dataset_sha256": json.loads((data_root / "meta.json").read_text())["sha256"]}
    (final / "train_meta.json").write_text(json.dumps(meta, indent=1, default=str) + "\n")
    if drive_dir.parent.exists():
        drive_dir.mkdir(parents=True, exist_ok=True)
        dst = drive_dir / "adapter"
        if dst.exists():
            shutil.rmtree(dst)  # our own previous copy of this run's adapter, never anything else
        shutil.copytree(final, dst, ignore=shutil.ignore_patterns("processor"))
        shutil.copy(log_csv, drive_dir / "train_log.csv")
        for f in ["train_meta.json"]:
            shutil.copy(final / f, drive_dir / f)
        size = sum(p.stat().st_size for p in drive_dir.rglob("*") if p.is_file())
        print(f"[train] adapter + log copied to {drive_dir} ({size / 1e6:.1f} MB)", flush=True)
    print(f"[train] done in {dt / 60:.1f} min; peak memory {peak:.1f} GB allocated / {reserved:.1f} GB reserved", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
