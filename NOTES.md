# NOTES

Findings and gotchas discovered while building. Measured, not assumed.

## Environment

Measured 2026-11-01 on Colab Pro (Step 1). Full machine-readable report: `results/env_report.json`.

- GPU: NVIDIA L4, 23034 MiB, compute capability 8.9, driver 580.82.07. 12 vCPU, 52 GB RAM, ~197 GB free on `/content`.
- System python is 3.13; uv 0.13.4 is preinstalled at `/usr/local/bin/uv`.
- **Python 3.12.3 (uv-managed) for all venvs.** Why: vLLM 0.31.0 declares `>=3.11,<3.15`, but 3.12 is the version vLLM, PyTorch and the CUDA wheel ecosystem (flash-attn, xformers, etc.) have the longest track record of prebuilt wheels on; 3.13 is newer and more likely to hit a missing wheel later. One version for every venv keeps behaviour identical.

| venv | key packages |
|---|---|
| `.venv-serve` | vllm 0.31.0, torch 2.13.0+cu130, transformers 5.17.0, compressed-tensors 0.17.0, openai 3.42.0, fastapi 0.136.3, pillow 13.0.0, pycocotools 2.0.11 |
| `.venv-train` | torch 2.13.0+cu130, transformers 5.17.0 (`Qwen3VLForConditionalGeneration` imports), peft 0.21.2, trl 1.17.1, datasets 5.1.0, accelerate 1.15.0 |
| `.venv-quant` | llmcompressor 0.14.0, compressed-tensors 0.19.0, torch 2.13.0+cu130, transformers 5.17.0 |

torch CUDA is available in all three (CUDA 13.0 runtime; the driver 580 supports it).

Install times (`make setup`):

| run | serve | train | quant | import verify | total |
|---|---|---|---|---|---|
| first ever, resolve from `.in` (uv cache partly warm from a failed attempt) | 19 s | 2 s | 2 s | 46 s | 69 s |
| second run, venvs exist (all skipped) | 8 s | 6 s | 7 s | 22 s | 43 s |
| cold: no venvs, `uv cache clean`, install from `.lock` (= a new Colab session) | 39 s | 3 s | 1 s | 55 s | 98 s |

So a new Colab session costs ~1.5-2 min of setup. Most of the "skip" and "verify" time is importing vLLM/torch, not installing.
- **Step 4:** strategies behind one registry (`search/base.py`); all text in `configs/prompts.yaml` (versioned, stored in every trace), all numbers in `configs/search.yaml`. Images always precede text. Eye search encodes the thumbnail once and asserts its hash is unchanged across fixations.
- **Step 5:** sliding window uses yes/no per tile (`tile_mode: confirm`); eye search keeps `retry_prompt: plain` and `refine_point: true`; the evidence label is off for evals. Downscale budget = R4's measured mean, 52,966 px. All chosen on dev (see Step 5 findings).

## Storage policy

Adopted 2026-11-01 (Step 3, Part A), when the user's Drive reached 14.5 / 15 GB. Paths are in `configs/base.yaml` and exported by `scripts/env.sh`; `make storage` prints the current usage.

| where | what | path |
|---|---|---|
| Colab local disk (wiped each session, ~175 GB free) | everything re-downloadable or regenerable with the fixed seed: HF model cache, COCO annotations, train images, generated training data, checkpoints, local quantized/merged models, logs | `/content/fvs_local/{hf_hub_cache,data,checkpoints,models}`, `logs/` |
| Google Drive (project budget ~500 MB, report after each step) | small irreplaceable files only: frozen eval images, LoRA adapter, copies of important logs | `MyDrive/foveated-vlm-search/{data/coco_val_eval,adapters}` |
| Hugging Face Hub (private repos) | merged fine-tuned model and its FP8 version (optionally the training dataset) | Steps 7-8 |

- `HF_HUB_CACHE=/content/fvs_local/hf_hub_cache`. `HF_HOME` is still not set, so the token stays at `~/.cache/huggingface/token`. `HF_HUB_DISABLE_SHARED_BLOBS=1` is kept.
- **Per-session model re-download, measured:** `hf download Qwen/Qwen3-VL-2B-Instruct` into the empty local cache took **12.9 s** for 4.26 GB (model.safetensors 4,255,140,312 B, sha256 = Hub etag, no pointer files). Cheap enough to repeat every session.
- Cleanup on 2026-11-01, approved by the user: deleted `hf_hub_cache/` (4.0 GB), `models/qwen3-vl-2b-instruct-fp8-dynamic/` (2.7 GB; regenerate with `python -m foveated_search.quant.fp8_dynamic` in about 21 s), `data/examples/` (174 KB) and the empty `checkpoints/` folder. Project Drive usage went from 6,796 MB to ~0 MB. The user has to empty the Drive trash in the browser before the quota frees up.
- Never delete from Drive or the HF Hub without asking (CLAUDE.md).

## Decisions

- **Version pins (2026-11-01).** Only releases at least two weeks old. vllm 0.32.0 (released 2026-10-18) was right on the edge and llmcompressor 0.15.0 was 3 days old, so we use vllm **0.31.0** (2026-10-05) and llmcompressor **0.14.0** (2026-09-22).
- **Separate `.venv-quant` for llm-compressor.** Installing `serve.in` with both packages failed in uv: `llmcompressor==0.14.0 depends on compressed-tensors==0.19.0 and vllm==0.31.0 depends on compressed-tensors==0.17.0`. torch and transformers ranges did overlap; the conflict is only compressed-tensors. Checked older llmcompressor releases: 0.13.0 pins compressed-tensors 0.18.0 and 0.12.0 pins 0.17.1. None match 0.17.0 exactly, so per the rules we did not force anything and split instead. `.venv-quant` keeps torch 2.13.0 and transformers 5.17.0 identical to `.venv-serve`.
  - **To verify in Step 8:** an FP8 checkpoint written by compressed-tensors 0.19.0 must load in vLLM with compressed-tensors 0.17.0. Test it with a tiny load before trusting R6.
- **Same torch (2.13.0+cu130) and transformers (5.17.0) in all three venvs**, so a model trained in `.venv-train` is processed the same way when served from `.venv-serve`.
- torch comes from plain PyPI (whose Linux wheel is cu130), not uv's `--torch-backend`, so vLLM's wheel and torch are built against the same CUDA.
- vLLM is installed first and torch recorded. Then the rest of `serve.in` is installed and the script aborts if torch changed. It didn't.
- (Superseded by the Storage policy in Step 3: `HF_HUB_CACHE` is now on the local disk.) **`HF_HOME` is not set**, so the HF token stays at `~/.cache/huggingface/token`.
- `scripts/env.sh` is generated by setup (paths only, no secrets) and is gitignored. `.lock` files are committed.
- The uv cache stays on local disk (default), not on Drive. Hardlinks across filesystems aren't possible, and Drive is slow for many small files. A cold install from the locks is only ~40 s anyway.
- **Step 2:** server flags all come from `configs/` via `foveated_search/serving/launch.py`; variants are overlay YAMLs (`configs/serve_fp8_online.yaml`, `configs/serve_fp8_offline.yaml`, selected with `FVS_CONFIG_OVERLAYS`).
- **Step 2:** point answers use vLLM structured outputs (json_schema), with a 100% parse rate. (Step 4: the schema is now `{"point_2d":[x,y]}` OR `{"absent":true}`; see Step 4 findings.) Image bounds are `min_pixels` 1024 / `max_pixels` 1228800. Clients use the stable served name `qwen3-vl`, whatever checkpoint is loaded. Details and evidence are under Findings.

## Findings

### Step 2: serving Qwen3-VL-2B-Instruct (vLLM 0.31.0, L4)

All numbers below come from `results/step2_*` and `logs/` of 2026-11-01.

**Model download and serving.**
- Download: `model.safetensors` is 4.26 GB (4,255,140,312 B, sha256 = Hub etag `7de1838c…`). `hf download` into the Drive `HF_HUB_CACHE` took 19 s (Xet). See Gotchas: the first download produced a broken pointer file, repaired and verified by sha256.
- Server startup (`make serve` to `/health` OK). First cold start: 145 s (weights 3.0 s, torch.compile 42 s, profiling + KV cache + warmup 96 s total). Later restarts: 63-100 s (compile cache warm). Online FP8: 163 s. Offline FP8 checkpoint from Drive: 184 s.
- GPU memory: the BF16 model takes 4.24 GiB. KV cache 12.34 GiB = 115,568 tokens (28x concurrency at 4096 ctx). nvidia-smi shows 17.5-19.2 GB in use because vLLM reserves `gpu_memory_utilization` 0.85 of 23 GB up front; `make serve-stop` brings it back to 0 MiB.
- **CUDA graphs: captured.** PIECEWISE (7 sizes) + FULL (2 to 5 sizes), capture sizes [1, 2, 4, 8, 16, 24, 32], graph pool 0.04 GiB, capture about 2 s. The vision encoder is not compiled or graphed (`compile_mm_encoder: False`).
- Latency is measured with `time.perf_counter` in the client on the same VM as the server (127.0.0.1). It is effectively server-side latency with no WAN round trip. TTFT = first streamed chunk with content.

**Vision tokens: client processor vs server agree exactly** (`results/step2_tokens.json`). For 5 sizes (96x72, 224x224, 640x480, 333x517, 1000x300), under both the default and the final pixel settings:
the processor's full prompt length == server `prompt_tokens`, and (prompt_tokens with image) - (text-only prompt_tokens) == processor vision tokens + 2 (`<|vision_start|>`, `<|vision_end|>`).
Token rule (patch 16, merge 2): resized sides snap to multiples of 32; tokens = (W/32)*(H/32).

**Coordinate format.**
- Docs: the model card only says "stronger 2D grounding". The official Qwen3-VL grounding cookbook (`cookbooks/2d_grounding.ipynb`) says: *"Qwen3-VL's default coordinate system has been changed from the absolute coordinates used in Qwen2.5-VL to relative coordinates ranging from 0 to 1000."* Native output keys are `point_2d` / `bbox_2d`.
- Measured (`results/step2_coords.json`, `results/step2_coords_synthetic.csv`): 36 synthetic queries (3 coloured shapes, 6 sizes/aspects, 2 random layouts each).
  - Median error 23 px if the output is read as relative 0-1000, vs 328 px if read as absolute pixels on the processor's resized image.
  - A linear fit of predicted vs true relative coordinates gives slope 1.00 (x) / 0.93 (y), R^2 0.97/0.96, at every size. That includes 96x72, which the default processor upscaled to 320x224 with a different scale per axis. So coordinates are **relative to the image as sent, independent of internal resizing**.
  - Residual error is an offset (points biased up-left, about -56/1000 in x overall, worst on small images), i.e. base-model imprecision, not a different convention. Hit rate (point inside the shape): 100% on 640x480, 480x640 and 1024x512; 50% on 96x72; 17% on 224x224 (n=6 each).
  - Real photo (COCO val 39769, `results/step2_point_check.png`): "remote control on the left" -> (88, 83) on the remote, correct. "cat on the right" -> (444, 156) on the cat, correct. "cat on the left" -> (143, 108) on the top edge of the cat's back, next to the remote: on target but borderline.
- Helper: `foveated_search/search/coords.py` (`model_to_px`, `px_to_model`, `model_to_frame`), unit-tested.

**Output format: vLLM structured outputs chosen** (`results/step2_format.json`). 24 calls each, 16 present + 8 negative ("purple star"), temperature 0:

| mode | parse rate | correct (present) | correct (negative) | median TTFT | median total | mean output tokens |
|---|---|---|---|---|---|---|
| JSON by prompt instruction | 75% | 50% | 100% | 74.7 ms | 318.5 ms | 16.4 |
| `response_format` json_schema | **100%** | **81%** | 100% | 76.5 ms | **290.5 ms** | 12.9 |

All prompt-mode failures were the model echoing both template options: `{"point": [264, 544], "point": null}`. `json.loads` would silently take the last key (null), so the parser now rejects duplicate keys as ambiguous. Structured output costs about +2 ms TTFT and is faster overall (no markdown fences). The schema is `POINT_RESPONSE_FORMAT` in `search/parsing.py`. The negative here is easy (no purple in the image), so 100% says little about false positives.

**Image resizing and token cost** (`results/step2_resize.csv`, 640x480 COCO frame; thumbnails keep 4:3, crops are square):

| image | sent | default: resized -> tokens | explicit (min 1024, max 1228800): resized -> tokens | PNG bytes |
|---|---|---|---|---|
| thumbnail 64 | 64x48 | 320x224 -> 70 | 64x64 -> 4 | 7,442 |
| thumbnail 96 | 96x72 | 320x224 -> 70 | 96x64 -> 6 | 15,841 |
| thumbnail 128 | 128x96 | 320x224 -> 70 | 128x96 -> 12 | 27,139 |
| thumbnail 192 | 192x144 | 320x224 -> 70 | 192x128 -> 24 | 58,703 |
| thumbnail 224 | 224x168 | 320x224 -> 70 | 224x160 -> 35 | 79,662 |
| crop 64 | 64x64 | 256x256 -> 64 | 64x64 -> 4 | 9,224 |
| crop 96 | 96x96 | 256x256 -> 64 | 96x96 -> 9 | 21,083 |
| crop 128 | 128x128 | 256x256 -> 64 | 128x128 -> 16 | 38,222 |
| crop 192 | 192x192 | 256x256 -> 64 | 192x192 -> 36 | 87,930 |
| crop 224 | 224x224 | 256x256 -> 64 | 224x224 -> 49 | 120,700 |
| full frame | 640x480 | 640x480 -> 300 | 640x480 -> 300 | 694,498 |

- Default `min_pixels` = 65,536 (from `preprocessor_config.json`) upscales **every** small image to at least ~256x256. Under defaults a smaller thumbnail saves no compute at all.
- Chosen: `image_min_pixels: 1024` (32x32 = 1 merged token, the true floor given patch 16 x merge 2) and `image_max_pixels: 1228800` (1280x960, so COCO frames and 720p webcam frames are never downscaled). Both are in `configs/base.yaml`, passed to vLLM as `--mm-processor-kwargs {"size": {...}}` and used by the client's token counter. After a restart the server's prompt_tokens matched the new counts for all 11 images.
- **For Step 4:** sides still snap to multiples of 32 (96x72 -> 96x64, 224x168 -> 224x160), so the model sees a slightly squashed thumbnail. Relative coordinates are unaffected. Thumbnail sides that are already multiples of 32 avoid the resample. "Pixels sampled" counts source pixels sent, per the definition.
- PNG bytes (compress level 6) are a weak bandwidth proxy for photos: the 96 px thumbnail is 15.8 KB vs 694 KB for the full frame (2.3%), while its pixel share is 2.25%.

**Latency, concurrency 1** (`results/step2_latency*.csv`; 1 warmup + 25 calls each; structured point prompt; 16 output tokens; prefix cache and mm-processor cache off):

| precision | image | vision tokens | TTFT p50 / p95 (ms) | total p50 / p95 (ms) |
|---|---|---|---|---|
| BF16 | thumbnail 96 (96x72) | 6 | 60.4 / 63.8 | 279.1 / 282.9 |
| BF16 | crop 224 | 49 | 67.2 / 69.9 | 286.4 / 288.8 |
| BF16 | full frame 640x480 | 300 | 101.3 / 102.6 | 323.6 / 325.2 |
| FP8 online | thumbnail 96 | 6 | 93.3 / 98.9 | 243.7 / 249.1 |
| FP8 online | crop 224 | 49 | 96.5 / 99.2 | 246.8 / 249.5 |
| FP8 online | full frame | 300 | 128.0 / 129.5 | 281.5 / 283.2 |
| FP8 offline (LM only) | thumbnail 96 | 6 | 66.7 / 71.4 | 216.6 / 221.5 |
| FP8 offline (LM only) | crop 224 | 49 | 70.5 / 78.8 | 221.0 / 228.8 |
| FP8 offline (LM only) | full frame | 300 | 101.4 / 102.8 | 255.5 / 257.3 |

- At batch 1, **decoding dominates**: BF16 runs about 14.6 ms per output token (memory-bound: about 4.2 GB of weights read per token), so 16 output tokens cost about 220 ms, while thumbnail vs full frame differs by only about 41 ms of TTFT. Fewer output tokens matter as much as fewer vision tokens. Keep the answer formats terse.
- FP8 cuts decoding to about 10 ms per token in both variants.
- Online FP8 raises TTFT by about 30 ms. It quantizes the vision tower too (see below). Hypothesis, not verified: the extra activation-quant kernels in the eager (non-graphed) vision encoder add launch overhead. Offline FP8, with the vision tower in BF16, keeps BF16-level TTFT.

**FP8 checks (de-risking Step 8).**
- **Online FP8 (`--quantization fp8`): works on the L4.**
  - vLLM 0.31 warns that `fp8` is deprecated for online use, in favour of `fp8_per_tensor`. It uses CutlassFP8ScaledMMLinearKernel and model memory drops from 4.24 to 2.56 GiB.
  - It quantizes all 216 Linear layers it finds: the language model (28 layers x qkv/o/gate_up/down), **and** the vision tower (24 blocks x attn.qkv/proj, mlp fc1/fc2), the 3 deepstack mergers and the visual merger. lm_head stays BF16.
  - The coordinate test passes: 36/36 parsed, hit rate 72.2% (same as BF16), median error 24 px (`results/step2_coords_fp8_online.json`).
- **Offline FP8_DYNAMIC (llm-compressor 0.14.0 in `.venv-quant` -> vLLM 0.31.0 in `.venv-serve`): LOADS.**
  - Recipe: `QuantizationModifier(targets=[Linear], scheme=FP8_DYNAMIC, ignore=["lm_head", "re:model.visual.*"])`, which quantizes the 196 language-model Linear layers. Data-free: load 6.7 s, quantize 3.3 s, save 11.3 s. The checkpoint was 2.85 GB on Drive (`models/qwen3-vl-2b-instruct-fp8-dynamic`; deleted in Step 3 under the storage policy, regenerable), format compressed-tensors / float-quantized.
  - vLLM loads it with `CompressedTensorsW8A8Fp8` + CUTLASS (2.93 GiB). So the compressed-tensors 0.19 (writer) vs 0.17 (vLLM) split from Step 1 is **not** a problem for this scheme.
  - Coordinate test: hit rate 72.7%, median error 24 px. But 3/36 prompt-JSON answers did not parse: one duplicate key and two runaway repetitions (`{"point": [...], "points": [{"point": ...`). Structured outputs would prevent both. **Step 8: rerun the format test on FP8.**
  - Which FP8 to use is decided in Step 8. Offline is faster (217 vs 244 ms per call) and keeps the vision tower in BF16, as CLAUDE.md specifies.

### Step 3: frozen evaluation set

**Frozen manifest: do not edit by hand.** Regenerate with `make eval-manifest` (seed 42, `configs/eval.yaml`, COCO `instances_val2017.json` from the zip with md5 `f4bbac642086de4f52a3fdda2de5fa2c`):

| file | sha256 |
|---|---|
| `results/eval_manifest.jsonl` (500 queries) | `c5aa81644b1268b34f59f691ce494159d26c0feca2be6258160580724b3f7489` |
| `results/eval_manifest_dev.jsonl` (50 queries) | `ceeecb5e01f2829a272b6eb6abf651a04a665a1a42d85e7870a1b1848673481b` |
| `results/eval_image_ids.txt` (500 ids, for the Step 6 no-overlap check) | `226aeb89660796bea8b65e6bb006371b7b667ee06422d16d3678c1c63fb5a9b9` |
| `results/eval_images_sha256.txt` (sha256 of each eval JPEG) | `49291a5ef0bd3ca2c5ed74a74cf287f68039ed4cc65655cedab17391c5e9bba3` |

Reproducibility, measured: 4 more runs of `make eval-manifest` in fresh processes (different `PYTHONHASHSEED`s) gave byte-identical manifest, dev, id and image-hash files. `test_regenerate_same_bytes` checks this on every `make test`, whenever the annotations are on local disk.

**Query design** (all values in `configs/eval.yaml`):
- 500 queries, one per image (500 distinct val2017 images). 400 positives / 100 negatives (20%).
- `food` (supercategory, 10 classes) makes up 22.5% of positives (90) and 22% of negatives (22). Single-category queries cover all 80 categories, drawn round-robin. Every category gets 3-4 positives (cap 6, never reached), and 78 categories get 1 negative.
- **Minimum size:** the target's largest non-crowd box must be ≥ 0.5% of the image. On a 640x480 frame that's ~39x39 px, which still leaves ~6x6 px in the 96x72 thumbnail; anything smaller is invisible at glance resolution and could only be found by a lucky fixation. Actual range: min 0.51%, median 7.3%, p90 50.6%.
- **Size buckets** (COCO definition, on the largest instance's segmentation area): small 18 (4.5%), medium 169 (42.2%), large 213 (53.2%). Some targets above the box-area floor are still COCO-"small" because their segmentation covers much less of the box than the box itself.
- **Ground truth** = all non-crowd boxes of the target ([x, y, w, h] in pixels, 2 decimals). For `food`, that's all boxes of all food classes. Instances per positive: mean 2.28, median 1, max 14.
- **Crowd rule (my addition, flagged to the user):** a positive is skipped if the target also has an `iscrowd` annotation in that image. Crowd regions are excluded from ground truth, so a correct point on one would score as a miss.
- **Negatives:** no annotation at all, crowd included, of any category in the target's supercategory (for `food`, no food at all). This guards against COCO's incomplete labels, e.g. a "cup" negative also contains no bowl, fork, bottle, etc.
- The dev subset (50) is stratified by (query_type, is_negative) with largest-remainder allocation: 40 pos / 10 neg, 11 `food`.

**Data handling** (storage policy):
- The annotations zip (252,907,541 B) is checked against the server's Content-Length and md5. `instances_val2017.json` (20 MB) and `instances_train2017.json` (470 MB, for Step 6) are extracted to `/content/fvs_local/data/coco/`, and the zip is then deleted from the local disk.
- The 500 eval images are fetched by URL with 8 workers and retries. Each is verified with a PIL decode and its size checked against the annotated width/height. They're written to Drive `data/coco_val_eval/` (**82.5 MB**) plus a local copy.
- Timings: first `make eval-manifest` took 66 s end to end (image fetch 45.6 s). Second run 8 s (all 500 skipped). A new session with Drive copies but no local ones: local copies restored from Drive in 1.1 s, no re-download.
- Project Drive usage after Step 3: **79 MB** (budget 500 MB).

### Step 4: search strategies

Code: `foveated_search/search/{base,strategies,geometry,schemas,viz}.py`, parameters in `configs/search.yaml`, prompts in `configs/prompts.yaml` (version 2). Live check: `make step4-check` -> `results/step4_traces.jsonl`, `results/step4_check.json`, `results/step4_trace_<query_id>.png`.

**Point format: measured, and the result overrides "shortest keys"** (`scripts/step4_point_format.py` -> `results/step4_point_format.json`; full frame, 50 dev queries = 40 positives + 10 negatives, BF16, temperature 0):

| answer format | hit rate (pos) | negatives answered absent | calibration slope x / y (pred vs true centre) | median offset x / y (0..1000) | out tokens pos / neg |
|---|---|---|---|---|---|
| `{"p":[x,y]}` / `{"p":null}` (prompts v1) | 42.5% | 9/10 | 0.74 / 0.64 | -165 / -111 | 13.9 / 6.8 |
| `{"point_2d":[x,y]\|null}` | 92.5% | 0/10 (never abstains) | 0.97 / 0.95 | -7 / 0 | 18.7 / 17.9 |
| native list `[{"point_2d":[x,y]}]` / `[]` | 42.5% (21/40 positives answered `[]`) | 10/10 | 0.83 / 0.84 | -19 / +9 | 10.6 / 2.0 |
| native list + label | 42.5% (21/40 `[]`) | 10/10 | 0.83 / 0.84 | -19 / +9 | 13.7 / 2.0 |
| `{"visible":bool,"point_2d":...}` | 90.0% | 10/10 | 1.03 / 0.97 | -7 / -20 | 22.2 / 21.1 |
| **`{"point_2d":[x,y]}` OR `{"absent":true}` (chosen, prompts v2)** | **92.5%** | 8/10 | **0.98 / 0.96** | **-6 / -3** | **17.1 / 8.9** |

- The key name decides grounding quality. With a non-native one-letter key the base model's points are compressed (slope < 0.75) and pulled up-left, which also explains Step 2's up-left bias (`{"point": ...}`). With Qwen3-VL's trained key `point_2d` and its own phrasing ("Locate the X ... report its point coordinates in JSON format"), points are centred and calibrated.
- The chosen anyOf costs ~3 more output tokens than `{"p":...}` on positives (~45 ms at BF16) and is cheap on "absent". The refined point in the confirm answer uses `point_2d` too. n = 50, so treat the exact rates as rough; Step 5 measures on 500.

**Structured-output whitespace (gotcha, fixed):** vLLM 0.31 ignores the per-request `structured_outputs.disable_any_whitespace` field (both the xgrammar and guidance backends read only the server config). Free whitespace produced at least one runaway answer of 100+ newlines. The flag is now server-level: `server.structured_disable_any_whitespace: true` plus `server.structured_backend: xgrammar` (the backend must be explicit, or startup fails: "disable_any_whitespace is only supported for xgrammar and guidance backends"). xgrammar in this mode still emits its standard separators (`{"point_2d": [78, 489]}`): no free whitespace, but `": "` and `", "` stay. The first, discarded run with whitespace allowed is kept as `results/step4_check_whitespace_allowed.json` (prompts v1).

**Output tokens per call type** (5 live-check queries, prompts v2, BF16; `results/step4_check.json`):

| strategy : call | n | mean completion tokens | mean call time |
|---|---|---|---|
| full_frame : point | 5 | 15.0 | 313 ms |
| downscale : point | 5 | 15.0 | 271 ms |
| sliding_window : tile | 5 | 17.0 | 299 ms |
| eye_search : point (thumbnail) | 6 | 13.7 | 244 ms |
| eye_search : confirm (crop) | 4 | 24.3 | 409 ms |

Confirm-field cost, measured by re-asking the same crops with each schema variant: a "no" is always 6 tokens. A "yes" costs 6 (`y` only), 12.7 (+ label), 22 (+ refined point), 30.3 (+ point + label). So the refined point costs ~16 tokens (~230 ms) and the label ~6.7 tokens (~100 ms).

**Observations from the traces (5 queries + a 50-query eye-search sweep; base BF16 model):**
- Sliding window answered "found" on the **first tile** in 5/5 queries, including the negative: on a 224 px tile the base model always points at something. With "stop at first hit" this is a false-positive machine; Step 5 will show its FP rate.
- Eye search never reached a 3rd fixation in the 50-query sweep (max 2 point calls, 0 repeats). After one rejected crop, the base model answers `absent` on the next thumbnail call.
- At batch 1, eye search's time-to-answer (578-857 ms on positives) is **higher** than full frame's (~320 ms): latency is decode-dominated, and a "yes" confirm with point + label is ~30 tokens. Turning the refined point and label off would save ~24 tokens (~350 ms) per found search. To decide in Step 5.
- The 24-character label cap truncates mid-word ("a banana split with a w/").
- Thumbnail byte identity held on all 12 multi-fixation searches of the sweep (asserted in code; one hash per search).
- Processor vision tokens == server prompt-token difference - 2 for every call of a 3-call eye search (96x64 -> 6, 224x224 -> 49).

**Pixel accounting on COCO (honest note).** A 96x64 thumbnail is 2.0% of a 640x480 frame, but one 224x224 fixation is 16.3%, so one fixation already costs more than eight thumbnails. On COCO-sized frames eye search samples ~18-30% of the frame per positive (the table above), which is more than the paper's budgets. The share shrinks with frame size: a 224 crop is 5.4% of a 1280x720 webcam frame and 2.4% of 1920x1080. `pixels_sampled_frac` counts every image sent (the thumbnail again on each fixation); `pixels_unique_frac` counts each distinct image once (what an edge device caching the thumbnail would send). Fixation size is configurable (`search.fixation_size`); a crop-size sweep is an open decision for Step 5.

### Step 5: eval harness and baselines R1-R4

Harness: `foveated_search/eval/` (splits, resumable runner, metrics, generated table). Named runs in `configs/runs.yaml`. Every run is `results/runs/<run>.jsonl` + `.meta.json` (config snapshot, prompts version, git commit, mode). `results/table.md` is generated by `make table`. Splits: dev 50 (all choices made here), test 450, all 500. Accuracy runs use concurrency 8; latency comes only from a seeded 100-query test subset at concurrency 1 after 3 warmup queries.

**Part A choices on dev** (50 queries; `results/table.md`, last section):

| dev comparison | found % | FP % (neg) | wrong loc. % | calls / query | output tokens | decision |
|---|---|---|---|---|---|---|
| sliding window, point per tile (Step 4) | 65.0 | **100.0** | 35.0 | 1.90 | 23.2 | kept as `tile_mode: point` |
| sliding window, yes/no per tile, point only on the "yes" tile | **87.5** | **0.0** | 10.0 | 5.06 | 38.9 | **default** (`tile_mode: confirm`) |
| eye search, retry prompt `plain` | 72.5 | 0.0 | 7.5 | 2.08 | 32.5 | **kept** (fewer calls/tokens, same accuracy) |
| eye search, retry prompt `encourage` | 72.5 | 0.0 | 5.0 | 2.16 | 33.4 | not adopted |
| eye search, refined point ON | **72.5** | 0.0 | 5.0 | 2.10 | 32.4 | **default** |
| eye search, refined point OFF (thumbnail point) | 52.5 | 0.0 | 25.0 | 2.10 | 22.5 | not adopted |

- Retry prompt: neither variant produced a new location after a rejected crop. With `plain` the model answers `absent`. With `encourage` it either answers `absent` or **repeats the exact same coordinates** (3 repeats on dev, e.g. `[143, 301]` twice). No dev search reached a 2nd crop with either prompt.
- Refined point confirmed on test: found 74.4% (on) vs 50.3% (off), wrong location 6.1% vs 29.7%. A thumbnail point is too coarse to land inside the box; the crop's refined point fixes that for ~16 output tokens.
- Evidence label is off for all evals (`search.confirm.label: false`). When on, the cap is 48 chars and labels that hit the cap are cut back to a word boundary.
- Run-to-run noise: `dev_eye_plain` and `dev_eye_refine_on` have identical settings but differ on 1 of 50 queries (wrong location 7.5% vs 5.0%). Batched requests at concurrency 8 change the GPU arithmetic slightly, so treat ±1 query on dev as noise.

**R1-R4 on test (450 queries; full table with CIs and breakdowns in `results/table.md`)**:

| row | found % [95% CI] | FP % [95% CI] | accuracy % | pixels % | vision tokens | output tokens | time-to-answer p50 / p95 (ms, conc. 1) |
|---|---|---|---|---|---|---|---|
| R1 full frame | 91.4 [88.3, 94.2] | 45.6 [35.6, 55.6] | 84.0 | 100.0 | 272 | 15.8 | 332 / 352 |
| R2 sliding window (yes/no tiles) | 81.7 [77.5, 85.8] | 7.8 [2.2, 14.4] | 83.8 | 101.7 | 274 | 42.3 | 648 / 1671 |
| R3 downscale (budget = R4 mean, 52,966 px) | 85.3 [81.7, 89.2] | 60.0 [50.0, 70.0] | 76.2 | 18.3 | 48 | 16.1 | 300 / 304 |
| R4 eye search (96 / 224, refined point) | 74.4 [70.0, 79.2] | 2.2 [0.0, 5.6] | 79.1 | 19.7 | 52 | 33.5 | 665 / 671 |

- The single-call strategies rarely say `absent`: full frame reports found on 45.6% of negatives, downscale on 60%. Eye search's separate confirm call on a sharp crop is what makes negatives work (FP 2.2%). At the same pixel budget, downscale finds more positives (85.3 vs 74.4%) but its accuracy is lower (76.2 vs 79.1%) because of false positives.
- At batch 1 the base eye search is about 2x slower than full frame (665 vs 332 ms p50). Decoding dominates: ~2.1 calls and 33 output tokens vs 1 call and 16 tokens. Its compute advantage (52 vs 272 vision tokens) doesn't show in latency on a 640x480 frame at batch 1.
- R4 is weakest on medium targets (63.2% found vs 84.2% large). On "food" it matches full frame (80.2%) with 0% FP.

**Sweeps (eye search, test, base BF16; `results/step5_sweeps.csv`, `results/step5_sweep_found_vs_pixels.png`)**:

| setting | found % | FP % | accuracy % | pixels % | vision tokens |
|---|---|---|---|---|---|
| thumb 64 / crop 224 | 61.1 | 1.1 | 68.7 | 18.0 | 47 |
| thumb 96 / crop 224 (R4) | 74.4 | 2.2 | 79.1 | 19.7 | 52 |
| thumb 128 / crop 224 | 78.1 | 1.1 | 82.2 | 22.5 | 59 |
| thumb 192 / crop 224 | **84.2** | 1.1 | **87.1** | 27.8 | 74 |
| thumb 96 / crop 160 | 63.9 | 1.1 | 70.9 | 11.6 | 31 |
| thumb 96 / crop 288 | 79.7 | 1.1 | 83.6 | 30.4 | 80 |

- Found % rises steadily with thumbnail size: the base model starts failing at small thumbnails (61% at 64 px, 74% at 96 px). The thumbnail matters more per pixel than the crop: going 96 -> 192 thumb costs +8.1 pp pixels for +9.8 pp found, and 224 -> 288 crop costs +10.7 pp pixels for +5.3 pp found.
- With a 192 px thumbnail, base eye search has the best overall accuracy of every configuration (87.1%, vs 84.0% for full frame) at 28% of the pixels and 27% of the vision tokens. FP stays at 1-2% at every setting.

**Failure analysis: R4 on test** (`scripts/step5_failures.py` -> `results/step5_failures.json`, 8 images in `results/step5_failures/`; categories assigned automatically from trace geometry, 20 failures checked by eye):

| category | count | what happens |
|---|---|---|
| off_target | 44 | thumbnail point lands on another object (cat -> garden bench, microwave -> top of fridge / range hood); the confirm correctly says no |
| in_crop_confirm_no | 13 | the crop contains the target (often small in the crop: spoons on a table) but the confirm says no |
| yes_point_off_in_crop | 13 | confirm says yes, refined point inside the crop misses every gt box |
| yes_wrong_object | 9 | confirm says yes on a crop without the target (a person under an umbrella for "cup") |
| absent_at_glance | 8 | first thumbnail call says absent (ball half cut off at the frame edge) |
| off_target_small | 5 | target < 8 px in the thumbnail (TV in a 96x32 panorama thumbnail) |
| false_positive (negatives) | 2 | confirm yes on a look-alike (orange-painted train front for "orange") |
| **total** | **94** (92 missed positives + 2 FP) | |

- Gave up after a rejected crop: **62 of 62** positives with a "no" never took a second crop. Those 62 are exactly the off_target + off_target_small + in_crop_confirm_no failures.
- **"food" ground-truth artifact.** COCO's "food" supercategory covers only 10 classes. Of the 12 "food" failures where the confirm said yes but the point missed the boxes, 9 land on real food COCO doesn't label (bread, rice, noodle salad, radishes, grilled chicken, pasta, stew, herbs, mushrooms), 2 are ambiguous, and 1 is a genuine miss (`results/step5_failures/review_food_yes_failures.png`). "food" found % understates the model; Step 6 must not train "no" on crops of unlabeled food.

**Where and why the base model fails (input for Step 6).** Half of all eye-search failures (49/94) are the glance itself: from a 96x64 thumbnail the base model points at a different, often similar-looking object (fridge top for microwave, bench for cat), or the target is only a few pixels wide. The second-biggest problem is that the search never recovers. In all 62 searches with a rejected first crop the model said "absent" (or repeated the same point) instead of proposing a second location, so max_fixations=3 is never used. On crops, the confirm call errs both ways: 13 times it rejects a crop that contains a small target, 9+2 times it accepts a crop without one (often colour or shape look-alikes). Once it does confirm, its refined point misses the boxes in 13 cases, most of them on "food", where most of the "misses" are real but unlabeled food. So the training data must teach (1) where-to-look on 64-192 px thumbnails, with similar-object distractors and small targets, (2) a second-best location after a rejected point, and (3) a sharper confirm on near-miss crops, plus a precise point inside the crop.

## Gotchas

- Colab sets `UV_SYSTEM_PYTHON=true` globally. That makes `uv venv` print a `--system has no effect` warning. `setup_colab.sh` unsets it, since we always target explicit venvs.
- `pycocotools` has no `__version__`; `check_env.py` falls back to `importlib.metadata`.
- Importing vLLM takes ~5-10 s, so even a no-op `make setup` takes ~40 s because it re-verifies imports.
- A bare `data/` in `.gitignore` also ignored the package folder `foveated_search/data/`. Data ignores are now root-anchored (`/data/`).
- **Hugging Face shared-blob store on Drive (huggingface_hub 1.33).** The first `hf download` left `snapshots/.../model.safetensors` resolving to a 79-byte regular file containing the text `../../blobs/53/530c…` (the per-repo blob should have been a symlink to the cache-wide "shared blob" store). vLLM / transformers would read 79 bytes. Plain symlinks and renamed symlinks on the Drive FUSE mount worked in a small test, so the exact cause was not reproduced (possibly Drive sync rewriting the link). Fix: `env.sh` exports `HF_HUB_DISABLE_SHARED_BLOBS=1` (classic per-repo blobs, which work). The existing cache was repaired by moving the 4.26 GB blob into place, and its sha256 matched the Hub etag. **Check `ls -lL` sizes after any new download to Drive.**
- **FlashInfer sampler needs a JIT build at warmup.** vLLM 0.31 uses FlashInfer top-k/top-p sampling by default, which JIT-compiles (needs `ninja`; "can take several minutes on first use") on every fresh VM. The first launch failed with `FileNotFoundError: 'ninja'`, because the venv's `bin` is not on PATH when `vllm` is called by full path. Fix: `server.flashinfer_sampler: false` (exports `VLLM_USE_FLASHINFER_SAMPLER=0`; we decode greedily anyway), and `serve.sh` puts the venv's `bin` on PATH for any other JIT.
- vLLM rejects per-request `mm_processor_kwargs` unless started with `--trust-request-mm-kwargs`. We keep that off and restart the server to change pixel limits.
- The model's `generation_config.json` sets temperature 0.7 / top_p 0.8 / top_k 20. The server runs with `--generation-config vllm` so only our request parameters (temperature 0, seed 42) apply.
- vLLM prefix caching is on by default in V1, and so is the mm-processor cache. Both are turned off in `configs/base.yaml` because repeating an identical benchmark image would otherwise skip work. R7 re-enables prefix caching on purpose.
- `collections.Counter.most_common()` breaks ties in insertion order, which here came from iterating a `set` of strings. Python randomizes string hashes per process, so the stats file changed between runs while the manifest didn't. Any ordering that feeds a committed file must be fully sorted (fixed: ties sorted by name).
- vLLM 0.31: per-request `structured_outputs.disable_any_whitespace` is silently ignored; only the server's `--structured-outputs-config '{"backend": "xgrammar", "disable_any_whitespace": true}'` takes effect (and needs an explicit backend). See Step 4.
- Plain `Image.quantize(256)` on a photo with coloured annotations maps the blue/purple markers to grey. `viz.save_png` reserves palette entries for the annotation colours.
- A bare `runs/` in `.gitignore` (meant for training-run folders) also ignored `results/runs/`. Anchored to `/runs/`, like `/data/` in Step 3.
- Markdown tables: labels containing `|` break the columns; row labels use commas.

