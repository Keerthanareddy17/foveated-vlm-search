---
license: apache-2.0
base_model: Qwen/Qwen3-VL-2B-Instruct
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

# qwen3-vl-2b-foveated-search

Qwen/Qwen3-VL-2B-Instruct fine-tuned (LoRA, merged into the BF16 weights) to **search for an object the way an eye does**:
glance at a small thumbnail of the whole frame, point where the object probably is, confirm on a sharp crop, and
look somewhere else after a wrong guess. It learned how to search across all 80 COCO object categories plus "food",
not a single class.

**Author:** Keerthana Reddy Katasani · **Code, write-up and results:** https://github.com/Keerthanareddy17/foveated-vlm-search · **Interactive replay:** https://keerthanareddy17.github.io/foveated-vlm-search/
· FP8 version: [honey613/qwen3-vl-2b-foveated-search-fp8](https://huggingface.co/honey613/qwen3-vl-2b-foveated-search-fp8)

## Results (from the project repository, `results/`)

- **Eye search on a 450-query COCO benchmark:** accuracy 79.1% (base model) → 85.6% with this
  fine-tune (+6.4 percentage points, paired 95% CI [+3.1, +10.0]), false positives on targets that are not in the
  image 1.1%. Full-frame use of the same model also learned to say "absent": false positives
  45.6% → 2.2%.
- **On a 1920x1440 photo** (FP8, one NVIDIA L4): found the queried food in 0.58 s with
  eye search vs 1.05 s looking at the whole frame, with 17.2x fewer vision tokens.
- **Honest caveat:** looking at the whole frame is still more accurate on COCO (94.2% vs 85.6% for eye
  search); eye search's advantage is cost (pixels, vision tokens, bandwidth), throughput on large frames and false alarms.

## What it was trained for

1. **Where to look:** from a small thumbnail of the whole frame (long side 64-192 px), point at the queried object,
   or answer that it is absent.
2. **Look again:** the same thumbnail plus already-checked locations; point somewhere else, or answer absent.
3. **Confirm:** on a sharp crop of the original frame, answer whether the object is there and, if so, point at it.

Points use Qwen3-VL's relative 0-1000 coordinates on the image as sent, in JSON: `{"point_2d": [x, y]}` or
`{"absent": true}`; confirm answers are `{"y": false}` or `{"y": true, "point_2d": [x, y]}` (optionally with a
short label `"l"`). The exact prompts are in `configs/prompts.yaml` in the repository.

## Training

35,497 examples (33,721 train, 1,776 validation) generated from COCO train2017
annotations, with no overlap with the evaluation images. LoRA r=16, alpha=32 on the
language model's linear layers; vision tower and mergers frozen. 1 epoch, effective batch
64, 527 steps in 31.5 min on one
NVIDIA L4; final validation loss 0.540.

## Use with vLLM

```bash
vllm serve honey613/qwen3-vl-2b-foveated-search --served-model-name qwen3-vl --max-model-len 6144 \
  --limit-mm-per-prompt '{"image": 1, "video": 0}' \
  --mm-processor-kwargs '{"size": {"shortest_edge": 1024, "longest_edge": 2764800}}'
```

`shortest_edge` 1024 keeps small thumbnails from being upscaled (a 96x64 thumbnail costs 6 vision tokens);
`longest_edge` 2764800 lets frames up to 1920x1440 through at native resolution. Ask with the project's prompt and
constrain the answer with structured outputs:

```python
import base64
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
image = base64.b64encode(open("thumbnail.png", "rb").read()).decode()
schema = {"anyOf": [{"type": "object", "properties": {"point_2d": {"type": "array", "items": {"type": "integer", "minimum": 0, "maximum": 1000}, "minItems": 2, "maxItems": 2}}, "required": ["point_2d"], "additionalProperties": False}, {"type": "object", "properties": {"absent": {"const": True}}, "required": ["absent"], "additionalProperties": False}]}
reply = client.chat.completions.create(
    model="qwen3-vl", temperature=0, max_tokens=24,
    messages=[{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image}"}},
        {"type": "text", "text": "Locate the food in the image with a point, report its point coordinates in JSON format. If there is no food, report {\"absent\": true}."}]}],
    extra_body={"structured_outputs": {"json": schema}})
print(reply.choices[0].message.content)   # {"point_2d": [x, y]} (0-1000, relative to the image) or {"absent": true}
```

The full search loop (thumbnail → point → sharp crop → confirm → look again) is in the project repository:
`foveated_search/search/strategies.py`.

## Limitations

Trained and evaluated on COCO images, plus a three-photo real-world demo. "food" means COCO's 10 food classes, so
real but unlabelled food (bread, rice, packaged sweets) counts as a miss or a false positive. Small targets are
limited by the thumbnail size.

## License

Apache-2.0, as the base model.
