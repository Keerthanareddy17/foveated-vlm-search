#!/usr/bin/env python
"""Step 7 memory check: one training step (forward + backward + AdamW step) on the 16 LONGEST
training samples, with the same LoRA model, collator and bf16 autocast as the Trainer.

Longest = exact token count from collate.encode. Candidates are pre-ranked with a fast estimate
(vision tokens from the processor's own resize rule + text tokens), then the top 200 are encoded exactly.
Writes results/step7_worst_batch.json. Run in .venv-train with vLLM stopped.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from foveated_search.config import REPO_ROOT
from foveated_search.train.collate import encode, load_processor
from foveated_search.train.train_lora import Collator, train_config

GIB = 1024**3


def main() -> None:
    from peft import LoraConfig, get_peft_model
    from PIL import Image
    from transformers import Qwen3VLForConditionalGeneration
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

    cfg = train_config()
    t = cfg["train"]
    root = Path(cfg.get_key("paths.data_dir")) / cfg.get_key("train_data.out_subdir")
    samples = [json.loads(line) for line in open(root / "train.jsonl")]
    proc = load_processor(cfg)
    lo, hi = int(cfg.get_key("server.image_min_pixels")), int(cfg.get_key("server.image_max_pixels"))
    t0 = time.perf_counter()

    def estimate(s: dict) -> int:
        m = s["meta"]
        h, w = smart_resize(m["sent_h"], m["sent_w"], factor=32, min_pixels=lo, max_pixels=hi)
        text = s["messages"][0]["content"][1]["text"] + s["messages"][1]["content"][0]["text"]
        return h * w // 1024 + len(proc.tokenizer(text)["input_ids"])

    ranked = sorted(range(len(samples)), key=lambda i: -estimate(samples[i]))[:200]
    exact = sorted(((len(encode(samples[i], root, proc)["input_ids"]), i) for i in ranked), reverse=True)[:16]
    idx = [i for _, i in exact]
    lengths = [n for n, _ in exact]
    print(f"[worst] 16 longest of {len(samples)} train samples: {lengths[0]}..{lengths[-1]} tokens "
          f"(types: {sorted({samples[i]['type'] for i in idx})}); ranking {time.perf_counter() - t0:.0f}s", flush=True)

    model = Qwen3VLForConditionalGeneration.from_pretrained(t["base_model"], dtype=getattr(torch, t["dtype"]),
                                                            attn_implementation=t["attn_implementation"]).cuda()
    model = get_peft_model(model, LoraConfig(r=int(t["lora"]["r"]), lora_alpha=int(t["lora"]["alpha"]),
                                             lora_dropout=float(t["lora"]["dropout"]),
                                             target_modules=t["lora"]["target_modules_regex"], bias="none",
                                             task_type="CAUSAL_LM"))
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(t["learning_rate"]))
    batch = Collator(proc.tokenizer.pad_token_id)([{k: v for k, v in encode(samples[i], root, proc).items()
                                                     if k != "n_prompt"} for i in idx])
    batch = {k: v.cuda() for k, v in batch.items()}
    torch.cuda.reset_peak_memory_stats()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = model(**batch).loss
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    reserved_now = torch.cuda.memory_reserved()
    overhead = (total - free) - reserved_now          # CUDA context etc., outside PyTorch's allocator
    peak_alloc, peak_res = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
    spare = total - (peak_res + overhead)
    out = {"batch_size": 16, "token_lengths": lengths, "padded_length": int(batch["input_ids"].shape[1]),
           "vision_tokens_in_batch": int(sum(g[0] * g[1] * g[2] // 4 for g in batch["image_grid_thw"].tolist())),
           "loss": round(float(loss), 4), "peak_allocated_gib": round(peak_alloc / GIB, 2),
           "peak_reserved_gib": round(peak_res / GIB, 2), "non_pytorch_overhead_gib": round(overhead / GIB, 2),
           "gpu_total_gib": round(total / GIB, 2), "spare_gib": round(spare / GIB, 2),
           "fits_with_1gib_spare": spare >= GIB}
    (REPO_ROOT / "results/step7_worst_batch.json").write_text(json.dumps(out, indent=1) + "\n")
    print("[worst] " + json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
