#!/usr/bin/env python
"""Model cards for the two Hugging Face repos, with numbers from results/ (same facts as the README).

  python scripts/build_model_cards.py     (make model-cards)  -> results/model_cards/{bf16,fp8}/README.md

Uploading is a separate, explicit step (make push-model-cards) that replaces only README.md in each repo.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_readme_tables import compute  # noqa: E402

from foveated_search.config import REPO_ROOT  # noqa: E402
from foveated_search.search.base import prompt  # noqa: E402
from foveated_search.search.base import search_config  # noqa: E402
from foveated_search.search.schemas import point_schema  # noqa: E402

OUT = REPO_ROOT / "results" / "model_cards"
AUTHOR = "Keerthana Reddy Katasani"


def common(F) -> dict:
    train = yaml.safe_load((REPO_ROOT / "configs/train.yaml").read_text())["train"]
    demo = yaml.safe_load((REPO_ROOT / "configs/demo.yaml").read_text())["demo"]
    hub = json.loads((REPO_ROOT / "results/step7_hub_push.json").read_text())
    fp8 = json.loads((REPO_ROOT / "results/step8_hub_push_fp8.json").read_text())
    cfg = search_config()
    return {"train": train, "github": demo["repo_url"], "pages": demo["pages_url"], "bf16_repo": hub["repo_id"],
            "fp8_repo": fp8["repo_id"], "prompt": prompt(cfg, "point", query="food"),
            "schema": json.dumps(point_schema())}


def results_block(F) -> str:
    v = F.v
    return f"""## Results (from the project repository, `results/`)

- **Eye search on a {v['test.n']}-query COCO benchmark:** accuracy {v['ft_acc.a']}% (base model) → {v['ft_acc.b']}% with this
  fine-tune ({v['ft_acc.diff']} percentage points, paired 95% CI {v['ft_acc.ci']}), false positives on targets that are not in the
  image {v['R5.fp']}%. Full-frame use of the same model also learned to say "absent": false positives
  {v['R1.fp']}% → {v['R1-ft.fp']}%.
- **On a {v['demo.positive2.food.frame']} photo** (FP8, one NVIDIA L4): found the queried food in {v['demo.positive2.food.eye.t']} s with
  eye search vs {v['demo.positive2.food.ff.t']} s looking at the whole frame, with {v['demo.positive2.food.tok_ratio']}x fewer vision tokens.
- **Honest caveat:** looking at the whole frame is still more accurate on COCO ({v['R1-ft.acc']}% vs {v['R5.acc']}% for eye
  search); eye search's advantage is cost (pixels, vision tokens, bandwidth), throughput on large frames and false alarms.
"""


def vllm_block(c: dict, repo: str) -> str:
    return f"""## Use with vLLM

```bash
vllm serve {repo} --served-model-name qwen3-vl --max-model-len 6144 \\
  --limit-mm-per-prompt '{{"image": 1, "video": 0}}' \\
  --mm-processor-kwargs '{{"size": {{"shortest_edge": 1024, "longest_edge": 2764800}}}}'
```

`shortest_edge` 1024 keeps small thumbnails from being upscaled (a 96x64 thumbnail costs 6 vision tokens);
`longest_edge` 2764800 lets frames up to 1920x1440 through at native resolution. Ask with the project's prompt and
constrain the answer with structured outputs:

```python
import base64
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
image = base64.b64encode(open("thumbnail.png", "rb").read()).decode()
schema = {c['schema'].replace('true', 'True').replace('false', 'False')}
reply = client.chat.completions.create(
    model="qwen3-vl", temperature=0, max_tokens=24,
    messages=[{{"role": "user", "content": [
        {{"type": "image_url", "image_url": {{"url": f"data:image/png;base64,{{image}}"}}}},
        {{"type": "text", "text": {json.dumps(c['prompt'])}}}]}}],
    extra_body={{"structured_outputs": {{"json": schema}}}})
print(reply.choices[0].message.content)   # {{"point_2d": [x, y]}} (0-1000, relative to the image) or {{"absent": true}}
```

The full search loop (thumbnail → point → sharp crop → confirm → look again) is in the project repository:
`foveated_search/search/strategies.py`.
"""


def bf16_card(F, c: dict) -> str:
    t, v = c["train"], F.v
    return f"""---
license: apache-2.0
base_model: {t['base_model']}
base_model_relation: finetune
library_name: transformers
pipeline_tag: image-text-to-text
language:
- en
tags:
- qwen3-vl
- visual-search
- grounding
- foveated
- lora
---

# {c['bf16_repo'].split('/')[-1]}

{t['base_model']} fine-tuned (LoRA, merged into the BF16 weights) to **search for an object the way an eye does**:
glance at a small thumbnail of the whole frame, point where the object probably is, confirm on a sharp crop, and
look somewhere else after a wrong guess. It learned how to search across all 80 COCO object categories plus "food",
not a single class.

**Author:** {AUTHOR} · **Code, write-up and results:** {c['github']} · **Interactive replay:** {c['pages']}
· FP8 version: [{c['fp8_repo']}](https://huggingface.co/{c['fp8_repo']})

{results_block(F)}
## What it was trained for

1. **Where to look:** from a small thumbnail of the whole frame (long side 64-192 px), point at the queried object,
   or answer that it is absent.
2. **Look again:** the same thumbnail plus already-checked locations; point somewhere else, or answer absent.
3. **Confirm:** on a sharp crop of the original frame, answer whether the object is there and, if so, point at it.

Points use Qwen3-VL's relative 0-1000 coordinates on the image as sent, in JSON: `{{"point_2d": [x, y]}}` or
`{{"absent": true}}`; confirm answers are `{{"y": false}}` or `{{"y": true, "point_2d": [x, y]}}` (optionally with a
short label `"l"`). The exact prompts are in `configs/prompts.yaml` in the repository.

## Training

{v['data.samples']} examples ({v['data.train']} train, {v['data.val']} validation) generated from COCO train2017
annotations, with no overlap with the evaluation images. LoRA r={t['lora']['r']}, alpha={t['lora']['alpha']} on the
language model's linear layers; vision tower and mergers frozen. {t['epochs']} epoch, effective batch
{int(t['per_device_batch_size']) * int(t['grad_accum'])}, {v['train.steps']} steps in {v['train.minutes']} min on one
NVIDIA L4; final validation loss {v['train.val_loss_final']}.

{vllm_block(c, c['bf16_repo'])}
## Limitations

Trained and evaluated on COCO images, plus a three-photo real-world demo. "food" means COCO's 10 food classes, so
real but unlabelled food (bread, rice, packaged sweets) counts as a miss or a false positive. Small targets are
limited by the thumbnail size.

## License

Apache-2.0, as the base model.
"""


def fp8_card(F, c: dict) -> str:
    v = F.v
    return f"""---
license: apache-2.0
base_model: {c['bf16_repo']}
base_model_relation: quantized
library_name: transformers
pipeline_tag: image-text-to-text
language:
- en
tags:
- qwen3-vl
- visual-search
- foveated
- fp8
- compressed-tensors
- llm-compressor
---

# {c['fp8_repo'].split('/')[-1]}

FP8 version of [{c['bf16_repo']}](https://huggingface.co/{c['bf16_repo']}), a Qwen3-VL-2B fine-tuned to search for an
object the way an eye does (glance at a thumbnail, point, confirm on a sharp crop, look again). This is the model
used in the project's photo demo.

**Author:** {AUTHOR} · **Code, write-up and results:** {c['github']} · **Interactive replay:** {c['pages']}

## Quantization

FP8_DYNAMIC with llm-compressor (FP8 weights with per-channel scales, FP8 activations with dynamic per-token scales,
no calibration data). Only the language model's linear layers are FP8; the vision tower, the vision-language mergers
and `lm_head` stay in BF16 ({v['fp8.bf16_modules']} modules). Checkpoint {v['fp8.ckpt_gb']} GB vs {v['bf16.ckpt_gb']} GB
in BF16. Made for GPUs with native FP8 (Ada, Hopper); vLLM loads it directly.

## Results (from the project repository, `results/`)

- **Same accuracy as BF16:** eye-search accuracy {v['fp8_acc.b']}% vs {v['fp8_acc.a']}% ({v['fp8_acc.diff']} percentage
  points, paired 95% CI {v['fp8_acc.ci']}) on a {v['test.n']}-query COCO benchmark.
- **{v['fp8.speedup_pct']}% faster** median time to answer at one request at a time on an NVIDIA L4
  ({v['R5.p50']} → {v['R6.p50']} ms); weights in vLLM {v['bf16.weights_gib']} → {v['fp8.weights_gib']} GiB.
- **On a {v['demo.positive2.food.frame']} photo:** found the queried food in {v['demo.positive2.food.eye.t']} s vs
  {v['demo.positive2.food.ff.t']} s looking at the whole frame, with {v['demo.positive2.food.tok_ratio']}x fewer vision tokens.
  Looking at the whole frame is still more accurate on COCO ({v['R1-ft.acc']}% vs {v['R5.acc']}%).

{vllm_block(c, c['fp8_repo'])}
## License

Apache-2.0, as the base model.
"""


def main() -> int:
    F, _ = compute()
    c = common(F)
    for kind, text in (("bf16", bf16_card(F, c)), ("fp8", fp8_card(F, c))):
        p = OUT / kind / "README.md"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        print(f"{p.relative_to(REPO_ROOT)} ({len(text):,} chars)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
