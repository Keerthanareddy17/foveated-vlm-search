"""FP8_DYNAMIC quantization with llm-compressor (run in .venv-quant; GPU must be free of vLLM).

  python -m foveated_search.quant.fp8_dynamic [--model MODEL_ID_OR_PATH] [--out DIR]

Recipe from configs (quant.*): FP8 weights + dynamic per-token FP8 activations on the
language model's Linear layers; vision tower and lm_head stay BF16. No calibration data.
Writes a compressed-tensors checkpoint that vLLM loads directly. Step 2 = compatibility check
on the base model; Step 8 reuses this for the fine-tuned model.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch


def main() -> None:
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    from foveated_search.config import REPO_ROOT, load_config

    cfg = load_config()
    q = cfg["quant"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(cfg.get_key("model.id")))
    ap.add_argument("--out", default=str(Path(cfg.get_key("paths.models_dir")) / q["output_name"]))
    ap.add_argument("--report", default="results/step2_fp8_offline_quantize.json")
    a = ap.parse_args()

    t0 = time.perf_counter()
    model = Qwen3VLForConditionalGeneration.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda")
    processor = AutoProcessor.from_pretrained(a.model)
    t_load = time.perf_counter() - t0

    recipe = QuantizationModifier(targets=list(q["targets"]), scheme=str(q["scheme"]), ignore=list(q["ignore"]))
    t1 = time.perf_counter()
    oneshot(model=model, recipe=recipe)
    t_quant = time.perf_counter() - t1

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t2 = time.perf_counter()
    model.save_pretrained(out, save_compressed=True)
    processor.save_pretrained(out)
    t_save = time.perf_counter() - t2
    # keep the processor / tokenizer / chat-template files byte-identical to the source model (as in the merge)
    import shutil

    from foveated_search.train.merge import COPY_FROM_BASE

    src = Path(a.model)
    if not src.exists():
        from huggingface_hub import snapshot_download

        src = Path(snapshot_download(a.model))
    copied = []
    for f in COPY_FROM_BASE:
        if (src / f).exists():
            shutil.copyfile(src / f, out / f)
            copied.append(f)

    size = sum(f.stat().st_size for f in out.glob("*.safetensors"))
    qc = json.loads((out / "config.json").read_text()).get("quantization_config", {})
    report = {
        "source": a.model, "out": str(out), "scheme": q["scheme"], "ignore": q["ignore"],
        "safetensors_gb": round(size / 1e9, 3), "load_s": round(t_load, 1), "quantize_s": round(t_quant, 1),
        "save_s": round(t_save, 1), "quant_method": qc.get("quant_method"), "format": qc.get("format"),
        "files_copied_from_source": copied, "quantized_layers_ignore": qc.get("ignore"),
    }
    print(json.dumps(report, indent=1))
    (REPO_ROOT / a.report).write_text(json.dumps(report, indent=1, default=str) + "\n")


if __name__ == "__main__":
    main()
