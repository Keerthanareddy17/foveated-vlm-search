---
license: apache-2.0
base_model: honey613/qwen3-vl-2b-foveated-search
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

# qwen3-vl-2b-foveated-search-fp8

FP8 version of [honey613/qwen3-vl-2b-foveated-search](https://huggingface.co/honey613/qwen3-vl-2b-foveated-search), a Qwen3-VL-2B fine-tuned to search for an
object the way an eye does (glance at a thumbnail, point, confirm on a sharp crop, look again). This is the model
used in the project's photo demo.

**Author:** Keerthana Reddy Katasani · **Code, write-up and results:** https://github.com/Keerthanareddy17/foveated-vlm-search · **Interactive replay:** https://keerthanareddy17.github.io/foveated-vlm-search/

## Quantization

FP8_DYNAMIC with llm-compressor (FP8 weights with per-channel scales, FP8 activations with dynamic per-token scales,
no calibration data). Only the language model's linear layers are FP8; the vision tower, the vision-language mergers
and `lm_head` stay in BF16 (105 modules). Checkpoint 2.85 GB vs 4.26 GB
in BF16. Made for GPUs with native FP8 (Ada, Hopper); vLLM loads it directly.

## Results (from the project repository, `results/`)

- **Same accuracy as BF16:** eye-search accuracy 85.1% vs 85.6% (-0.4 percentage
  points, paired 95% CI [-2.0, +1.1]) on a 450-query COCO benchmark.
- **24% faster** median time to answer at one request at a time on an NVIDIA L4
  (667 → 507 ms); weights in vLLM 4.24 → 2.93 GiB.
- **On a 1920x1440 photo:** found the queried food in 0.58 s vs
  1.05 s looking at the whole frame, with 17.2x fewer vision tokens.
  Looking at the whole frame is still more accurate on COCO (94.2% vs 85.6%).

## Use with vLLM

```bash
vllm serve honey613/qwen3-vl-2b-foveated-search-fp8 --served-model-name qwen3-vl --max-model-len 6144 \
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

## License

Apache-2.0, as the base model.
