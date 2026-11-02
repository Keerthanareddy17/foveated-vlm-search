# NOTES

Findings and gotchas discovered while building. Measured, not assumed.

## Resuming in a new session (state after Step 8b, 2026-11-01)

Everything needed for Steps 9-10 is in git (code, configs, results), on Drive (eval images, LoRA adapter, logs) or on the HF Hub (models). The Colab local disk can be wiped at any time.

**Order of commands in a fresh Colab runtime:**

1. Notebook setup cell: mount Google Drive at `/content/drive` and log in to Hugging Face (token at `~/.cache/huggingface/token`; it must be able to read the private repos below). Then clone the repo into `/content/foveated-vlm-search`.
2. `make setup`: recreates `.venv-serve`, `.venv-train`, `.venv-quant` from the lock files and writes `scripts/env.sh`. Measured cold: **~1.5-2 min** (98 s in Step 1).
3. `make check`: GPU, venvs, Drive, HF token, disk: expect all PASS.
4. `make serve-demo`: serves the fine-tuned FP8 model straight from the private Hub repo `honey613/qwen3-vl-2b-foveated-search-fp8` with native-resolution frames (`configs/serve_hires.yaml`) and the demo search settings (`configs/demo.yaml`: 288 px thumbnail, look-again mask on). First start downloads the model (2.86 GB) into `/content/fvs_local/hf_hub_cache`: measured **178 s from empty cache to READY**, download included. Then `make demo-smoke` runs one eye-search query with the demo settings (measured: correct, found on the 3rd fixation of a 1280x960 frame, 1.9 s), and `make serve-stop` stops it.

**Re-downloaded or recreated automatically:** the venvs (from `requirements/*.lock`, ~1.5 min), the FP8 demo model (2.86 GB, on first `make serve-demo`), the local copies of the 500 eval images (`make eval-manifest` restores them from Drive in ~1 s; the eval code also reads the Drive copies directly). The COCO annotation zip (253 MB, ~15 s) is only fetched if `make eval-manifest` or `make gen-data` runs.

**NOT needed for Steps 9-10** (regenerable, so don't recreate unless an earlier step is re-run): COCO train2017 images (1.2 GB, `make gen-data` refetches them), the generated training data (2.0 GB, seeded, `make gen-data`), the 2x2 / 3x3 mosaics (2.1 GB, `make mosaics` in ~30 s), the base model cache (4.3 GB; only for re-running base-model rows), the local merged BF16 model (also on the Hub: `honey613/qwen3-vl-2b-foveated-search`), LoRA training checkpoints (the final adapter is on Drive: `adapters/lora_r16_e1/`), and the local FP8 copy (on the Hub).

**Where things live:** results/runs/tables/figures in git; Drive `MyDrive/foveated-vlm-search/` = eval images (79 MB) + LoRA adapter and training log (67 MB) + this session's logs (`logs/2026-11-01/`, 4 MB) = 150 MB of the 500 MB budget; HF Hub (private) = merged BF16 model and its FP8 version.

**Step 9 demo pipeline (after `make serve-demo`):** put the phone photo in `test_pictures/`, then `make intake` (upright, metadata-free 1920 px copy in `test_pictures/clean/`), `make demo IMAGE=test_pictures/clean/<name>.jpg QUERY="food"` (or list pairs in `configs/demo_queries.yaml` and `make demo-batch`), then `make replay` -> `results/demo/replay.html`. Originals in `test_pictures/` are gitignored.

**Demo defaults to remember (from Step 8b):** large frames (720p and up) are where eye search pays off; use the 288 px thumbnail and `search.look_again_mask: true` (both in `configs/demo.yaml`). R1-R7 keep the frozen Step 5 settings (96 px, mask off).

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
- **Step 6:** training data shares code with inference (`search/views.py`, `point_prompt` / `confirm_prompt`, `dump_point` / `dump_confirm`, `encode_png`); confirm labels use mask-based visibility; "food" negatives exclude 13 food-adjacent classes.
- **Step 7:** LoRA r16 on the decoder only, 1 epoch, effective batch 64 (16 x 4); final adapter used (val loss = best); merged model ships the base config/processor files byte-for-byte; private HF repo `honey613/qwen3-vl-2b-foveated-search`.
- **Step 8:** prompts and search settings unchanged; R5-R7 differ only in the served model / server flags (overlays). R3-ft budget = R5 mean (50,026 px). Images are encoded off the event loop (`VLMClient.aencode`). W4A16 skipped (reason under Step 8).
- **Step 8b:** high-resolution mosaics from the frozen eval set (308 queries that fit a 640x480 cell); hi-res server overlay (max pixels 1920x1440, max_model_len 6144); `search.look_again_mask` added (helps, +4.4 pp accuracy; default off to keep R1-R7 reproducible, recommended on from Step 9).
- **Step 9:** scope changed by the user: one (maybe two) phone photos with several queries, no desk set, no labeling tool, no live camera. All demo settings in `configs/demo.yaml` (search overlay + `demo:` section). Evidence label turned **on** for the demo only (`search.confirm.label: true`; trained on 9.8% of confirm samples), off for every eval row. Clean copies are baseline JPEG q92 4:4:4 with no metadata (small enough to commit; the model gets a lossless PNG of the decoded pixels either way).

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

### Step 6: training data

`make gen-data` (plan -> fetch -> generate -> stats + grid -> `.venv-train` processor check). Config `configs/train_data.yaml`, code `foveated_search/data/{train_data,masks,train_data_report}.py`, `foveated_search/train/{collate,check_data}.py`. Data on the local disk only: `/content/fvs_local/data/train_data/` (2.0 GB, 35,497 PNGs + jsonl + meta.json) and the source images in `/content/fvs_local/data/coco_train/` (1.2 GB). Nothing on Drive or the Hub.

**Frozen dataset** (seed 42, prompts v3; regenerated in fresh processes, byte-identical jsonl and PNGs):

| file | sha256 |
|---|---|
| `train.jsonl` (33,721 samples) | `c5d0ae5be1e67245d9e07a0b4505f6d30a64aa4342d9687adc0ef9fd1312b0c7` |
| `val.jsonl` (1,776 samples) | `01aec5ba7950f6439c6b1ab216bd6874fd1d71a441b298075e76863a0756ad9e` |

**Train/inference consistency (how it is guaranteed, not just checked):**
- Thumbnails and crops come from `search/views.py` (`make_thumbnail`, `make_crop`), and eye search was refactored to call the same functions. Prompt text comes from `search/base.py` (`point_prompt`, `confirm_prompt`, `checked_text`), which eye search also uses. A test asserts the mock eye search's request texts equal the generator's.
- Targets come from `schemas.dump_point` / `dump_confirm` (`json.dumps` separators `", "` / `": "` = xgrammar's output under our server setting). Every target in the dataset round-trips through `parse_p` / `parse_confirm` (test).
- PNG bytes come from the client's own encoder (`serving.images.encode_png`, level 6).
- `.venv-train` processor with `size = {shortest_edge: 1024, longest_edge: 1228800}` (same as the server): **all 305 checked 96x64 thumbnails = 6 vision tokens**; 0 of 1,200 grid-aligned images resized; 0 of 1,200 trained spans differed from `target + <|im_end|>` (`results/step6_token_stats.json`). The chat template adds no system prompt and no thinking block; the prompt part (`<|im_start|>user ... <|im_start|>assistant\n`) is masked, only the JSON + `<|im_end|>` is trained, and the trailing newline is masked.

**Source images:** 100 images per category from train2017 (seeded) -> 7,768 selected; the final plan uses 7,516 (376 val, split by image). Zero overlap with the eval image ids (asserted while planning and in a test). First fetch: 7,541 images, 1,224 MB, 91 s with 32 workers (~80 img/s), all verified (PIL decode + size). Later plan revisions fetched 177 + 45 more; re-runs verify and skip all of them in ~7 s.

**Samples** (`results/step6_dataset_stats.md`): 35,497 = where 13,999 (25% absent) + look_again 5,816 (15% absent) + confirm 15,682 (51% yes / 49% no); train 33,721 / val 1,776. 503 planned jobs dropped for lack of valid geometry (318 confirm, 184 look_again, 1 where). "food" is 18% of every type; the 80 single categories are balanced by round-robin. Thumbnail long side 64/96/128/160/192 = 14/40/15/15/15%; crops 160/224/288 = 14/70/16%. Positives/yes below the eval's 0.5% minimum area: 18% (Type 1/1b), 5% (confirm). "No" crops: distractor 48%, background 33%, near miss 19%. 870 "yes" crops show part of an object bigger than the crop. Evidence label on: 9.8% of confirm prompts.

**Tokens per sample** (processor in `.venv-train`, 400 per type): where mean 69.9 / p95 84 (11.5 vision, 14.7 trained); look_again 102.6 / 122 (11.7 vision, 15.1 trained); confirm 114.9 / 152 (50.0 vision, 15.0 trained). About **3.2M tokens per epoch** (33.7k train samples). Input for Step 7's time estimate.

**Design fixes made while reviewing the first generations (all on the training data only):**
- On-object target points: mask centroid if inside the mask, else the pixel farthest from the mask border (erosion). On a donut the point lands on the ring, not in the hole.
- "Yes" crop visibility is measured on the instance MASK: >= 50% of the mask inside the crop, or the visible part covers >= 25% of the crop. With box-based visibility, objects bigger than the crop could never be a "yes" (many dropped jobs), so the model would never have seen "the crop is all object". The yes point is the on-object point of the visible part.
- Near-miss "no" crops are planned on images that contain the target (non-food); the first version picked the kind after the image and produced only 0.9% near misses (now 19%).
- Only the `pos_small` variant may use targets below the eval's 0.5% minimum. The first version applied the 0.1% floor to every positive (35-41% COCO-small).
- "food": negatives and "no" crops only from images with no food class and none of 13 food-adjacent classes (bowl, cup, dining table, ...). Look-again checked points for "food" also stay outside those classes' boxes, so "no food there" is never said about a bowl or a table. Residual risk: an image with no labelled food or food-adjacent object can still show unlabelled food (seen once in the grid: a baby eating something red).

**"food" scores in the eval understate the model:** Step 5 found 9 of the 12 "food" confirm-yes location misses landed on real food that COCO doesn't label (bread, rice, noodles, stew, ...). The eval keeps COCO's 10 classes as ground truth, so a fine-tuned model that points at unlabelled food will still be scored as a miss.

**Base model on val samples (before training; `results/step6_base_on_val.json`, 300 per type, same parsers, BF16):**

| type | success | positives / yes | negatives / no | notes |
|---|---|---|---|---|
| where (Type 1) | 42.7% | 42.1% | 45.0% | points at something on 55% of negatives |
| look_again (Type 1b) | 15.0% | **0.0%** | 100% | answers `{"absent": true}` on all 255 positives |
| confirm (Type 2) | 90.0% | 84.4% | 95.4% | near-miss "no": 85.2%; distractor 97.4%; background 97.9% |

The confirm skill is already decent, Type 1 is weak on small thumbnails and negatives, and look-again is entirely missing: the "never recovers" failure from Step 5, measured directly.

### Step 7: LoRA fine-tuning, merge, push

Code: `foveated_search/train/{train_lora,merge,push_hub}.py`, config `configs/train.yaml`; `make train-bench`, `make train`, `make merge`, `make push-hub`.

**Setup.** Base Qwen3-VL-2B-Instruct BF16 (sdpa attention). LoRA r=16, alpha=32, dropout 0.05 on all 196 decoder projections (q/k/v/o + gate/up/down x 28 layers); vision tower and mergers frozen (asserted at startup). 17,432,576 trainable / 2,144,964,608 (0.81%). AdamW, lr 2e-4, cosine, 3% warmup, grad clip 1.0, seed 42. Effective batch 64 = 16 x 4, no gradient checkpointing. Plain transformers `Trainer` (no TRL), inputs only from `train/collate.py`. At startup the Trainer's dataloader batch for the first 64 train samples (no shuffle) is checked to equal our collator's output tensor for tensor: True.

**Gotchas found:** transformers 5.17 has no `warmup_ratio` (`warmup_steps` takes a float < 1 as a ratio). Qwen3-VL in transformers 5.x also needs the processor's `mm_token_type_ids` for multimodal RoPE, so it is passed through `encode` and padded with 0. The Trainer evaluates at the final step on its own, so the last val point is the final adapter.

**Memory and speed (measured).** Batch 32 per device ran out of memory (20.5 GB allocated; logits over the 151k vocabulary). The 30-step benchmark at 16 x 4: 17.3 samples/s, 19.9 GB allocated / 21.3 GB reserved (`results/step7_bench.json`). Worst-case check on the 16 longest training samples (288 px confirm crops, 168-170 tokens), one forward + backward + AdamW step: 18.49 GiB allocated / 19.98 GiB reserved + 0.25 GiB CUDA overhead of 22.03 GiB, so **1.81 GiB spare** (`results/step7_worst_batch.json`). Eval batch 16 (64 would add ~6 GB of logits). `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

**Run.** 1 epoch = 527 steps, **31.5 min** (17.9 samples/s), peak 20.1 GB allocated / 21.5 GB reserved, no NaN, no OOM, no resume needed. Train loss 1.78 (step 5) -> ~0.55 by step 20 -> ~0.49 at the end (mean over the run 0.526). Val loss on the full val split (1,776 samples) at steps 53/106/159/212/265/318/371/424/477/527: 0.562, 0.575, 0.580, 0.556, 0.550, 0.547, 0.548, 0.546, **0.540, 0.540** (`results/step7_train_log.csv`, `results/step7_loss_curves.png`).
- Val loss rose twice early (steps 106 and 159) while train loss eased; this was flagged during the run. Likely cause: the model fits the training images (~4.7 samples per image) faster than it generalizes, while the learning rate is still near its peak. It reversed once the cosine decay set in, and val loss improved steadily to the end. The final adapter equals the best val loss (0.5405 vs 0.5403 at step 477), so it is used.
- Final train/val gap ~0.05. A second epoch looks unnecessary: val loss had flattened (0.546 -> 0.540 over the last 2 evals).

**Artifacts.** Adapter on Drive `adapters/lora_r16_e1/adapter/` (69.8 MB) + `train_log.csv` + `train_meta.json`. Project Drive usage after Step 7: 146 MB. Merged model on the local disk (`/content/fvs_local/models/qwen3-vl-2b-foveated-search`, 4.26 GB) and on the Hub: **private** repo `honey613/qwen3-vl-2b-foveated-search` (token role checked: write; 4.27 GB, `results/step7_hub_push.json`).

**Merge checks** (`results/step7_merge_diff.json`): all 315 vision-tower/merger tensors bit-identical to the base; all 196 decoder projection weights changed; tensor key sets identical. Processor, tokenizer, chat template, preprocessor and generation configs copied byte-for-byte from the base. transformers 5.17 re-serializes `config.json` differently (`rope_scaling` + `rope_theta` -> `rope_parameters` with the same values, added `dtype` fields, `transformers_version` 4.57 -> 5.17, vision `model_type` renamed). The architecture is identical, so **the base `config.json` is shipped byte-for-byte** (zero diff). vLLM 0.31 loads the merged model with the same server config.

**Before / after on the same 900 val samples** (Step 6 scorer, BF16; `results/step7_checks.md`):

| skill | subset | base | fine-tuned |
|---|---|---|---|
| where (first glance) | positives | 42.1 | 52.1 |
| where (first glance) | negatives (absent) | 45.0 | **90.0** |
| look again | positives | **0.0** | **50.2** |
| look again | negatives (absent) | 100.0 | 97.8 |
| confirm | yes crops | 84.4 | 90.5 |
| confirm | no crops | 95.4 | 96.1 (near miss 85.2 -> 85.2) |

No negative subset dropped by more than 5 pp: fine-tuning did not make the model over-eager.

**Dev sanity runs (50 queries, accuracy mode; `results/runs/dev_*_finetuned.jsonl`):**
- Eye search: found 72.5% -> **85.0%**, FP 0% -> 0%, accuracy 78% -> 88%, calls/query 2.08 -> 1.82. Only 2 searches (was 11) had a rejected crop, and 1 of the 2 took a second fixation (base: 0 of 11). Too few to judge the look-again skill on dev; Step 8 on test will.
- Full frame (forgetting check): found 92.5% -> 97.5%, FP 30% -> **0%**. No forgetting; the "absent" answer now transfers to the full frame too.
- Caveats: dev is small (50 queries) and was used for every Step 5 choice; the test numbers in Step 8 are the real ones.

### Step 8: fine-tuned eval, FP8, prefix caching, serving (R5-R7)

Same harness, splits and frozen Step 5 settings (96 px thumbnail, 224 crop, refined point on, label off, prompts v3); only the served model / server flags change, via overlays (`configs/serve_finetuned*.yaml`, named runs in `configs/runs.yaml`). Full table: `results/table.md`. Accuracy at concurrency 8; latency only from the 100-query latency subset at concurrency 1.

**Test split (450), with 95% bootstrap CIs:**

| row | found % | FP % | accuracy % | pixels % | vision tok | time-to-answer p50 / p95 (ms) |
|---|---|---|---|---|---|---|
| R4 base, BF16, eye search | 74.4 [70.0, 79.2] | 2.2 | 79.1 | 19.7 | 52 | 665 / 671 |
| R5 fine-tuned, BF16, eye search | 82.2 [78.3, 86.1] | 1.1 | 85.6 | 18.5 | 49 | 667 / 1021 |
| R6 fine-tuned, FP8, eye search | 81.7 [77.5, 85.8] | 1.1 | 85.1 | 18.4 | 49 | 507 / 797 |
| R7 = R6 + prefix + mm cache | 81.7 [77.5, 85.8] | 1.1 | 85.1 | 18.4 | 49 | 508 / 748 |
| ref R1-ft fine-tuned, full frame | **93.3** [90.6, 95.8] | 2.2 | **94.2** | 100 | 272 | 332 / 346 |
| ref R3-ft fine-tuned, downscale (budget = R5 mean, 50,026 px) | 89.7 [86.7, 92.8] | 6.7 | 90.4 | 18.0 | 47 | 301 / 305 |
| ref R4-192 base, eye search, 192 px thumbnail | 84.2 [80.3, 87.8] | 1.1 | 87.1 | 27.8 | 73 | n/a |

**What fine-tuning bought (paired, same test queries):**
- Eye search R4 -> R5: found **+7.8 pp [+4.2, +11.7]**, accuracy +6.4 pp [+3.1, +10.0]; 42 queries wrong->right, 13 right->wrong. FP 2.2% -> 1.1%, at slightly fewer pixels (18.5 vs 19.7%) and calls (2.05 vs 2.15).
- The fine-tuned sweep is above the base at every thumbnail size (`results/step8_sweep_ft.csv`, `results/step8_sweep_base_vs_ft.png`): 64 px 61.1 -> 73.9, 96 px 74.4 -> 82.2, 128 px 78.1 -> 84.7, 192 px 84.2 -> 88.1 (accuracy 90.2). The fine-tuned model at 64 px matches the base at 96 px; FP 0-1.1% throughout.
- **Fine-tuning helped the single-call strategies even more.** Full frame went from 45.6% to 2.2% FP (accuracy 84.0 -> 94.2), because the "where" training taught it to answer `absent`. Once both use the fine-tuned model, **full frame beats eye search**: R1-ft vs R5 accuracy **-8.7 pp [-11.8, -5.1]** for eye search (found -11.1 pp; 50 right->wrong vs 11 wrong->right). At the same pixel budget, downscale (R3-ft, 90.4%) also beats eye search (85.6%). On 640x480 COCO frames, a 2B model looking once at a moderately downscaled frame is better than two sharp glimpses.

**"Look again" in real searches** (test, `results/table.md` second-fixation section): R4 had 112 searches with a rejected crop and took 0 second fixations. R5 had 49 (41 on positives); 19 took a second fixation (14 on positives), which found the target 5 times with 0 false positives; 3 reached a third fixation. But R5 logged **70 repeats**: after a rejection, the model's next point often falls back inside the 224 px crop it already checked. The training data taught "point somewhere other than the listed checked point", while eye search treats any point inside the checked *crop* as a repeat (train/inference mismatch: the look-again checked points should be crop-sized exclusion zones). The skill exists but delivers little: ~5 extra finds out of 360 positives.

**What FP8 cost and saved** (R5 -> R6, `results/step8_fp8_quantize.json`, `results/step8_val_threeway.md`):
- Accuracy: **-0.4 pp [-2.0, +1.1]** (found -0.6 pp [-2.5, +1.1]): no measurable cost. Parse failures: **0** over 1,220 R6 calls (all + latency) and 0 over 900 FP8 val-sample calls. Val samples: within 1-2 pp of fine-tuned BF16 everywhere, near-miss "no" unchanged (85.2%). Exception: look-again negatives 97.8 -> 91.1% (3 of 45 samples).
- Quantization (llm-compressor FP8_DYNAMIC, decoder only; 105 vision/merger/lm_head modules kept in BF16): 11 s (load 2.0, quantize 0.9, save 8.4); checkpoint 2.85 GB vs 4.26 GB; vLLM weights 2.93 vs 4.24 GiB; KV cache 142,656 vs 131,696 tokens (+8%). Private repo `honey613/qwen3-vl-2b-foveated-search-fp8`; model on the local disk only, not on Drive.
- Latency: time-to-answer p50 667 -> 507 ms (-24%). Per call at concurrency 1: thumbnail 294 -> 224 ms, crop 375 -> 281 ms, full frame 332 -> 261 ms. Decoding at batch 1 is memory-bound, and FP8 halves the weight bytes.

**What prefix caching did** (R7, `results/step8_prefix_cache.json`, `results/step8_r7_metrics.txt`): flags `--enable-prefix-caching --mm-processor-cache-gb 4 --enable-prompt-tokens-details`.
- Token hit rate **5.6%** on the test run (3,712 / 66,527 prompt tokens; cold cache), 5.3% on the latency subset (cold). Image-preprocessing cache: 10.2% of images.
- **Every hit is a thumbnail call on fixations 2-3**: all 94 such calls hit, reusing 48% of their prompt tokens (the byte-identical thumbnail + prompt prefix). Fixation-1 thumbnails and crop confirms: 0 hits (the image comes near the start of the prompt, so different requests share less than one 16-token KV block).
- Latency at concurrency 1: p50 508 vs 507 ms, p95 748 vs 797 ms: no measurable effect, as expected: only 10% of calls benefit and decoding dominates.
- Accuracy identical on average (85.1%), but 14 queries flipped (7 each way): reused KV blocks give slightly different numerics, the same kind of run-to-run noise as batching.
- **Measurement trap:** the first R7 latency run came right after the R7 test run on the same server and showed a 50.9% hit rate, because the latency subset is part of test. Its cache was hot with identical requests. That run was discarded and re-run after a server restart (cold cache).

**Serving under load** (`results/step8_serving.csv`, `results/step8_throughput.png`; one L4, 100-query latency subset):

| completed searches / s | c=1 | c=4 | c=8 | c=16 |
|---|---|---|---|---|
| R1-ft full frame, BF16 | 2.52 | 7.46 | 11.21 | 14.63 |
| R5 eye search, BF16 | 1.62 | 4.96 | 8.51 | 13.24 |
| R6 eye search, FP8 | 2.12 | 5.84 | 9.40 | 14.35 |
| full frame, FP8 (reference) | 2.99 | 8.88 | 12.86 | 16.45 |

- On 640x480 frames, **full frame serves more searches per GPU at every concurrency**, BF16 vs BF16 and FP8 vs FP8. Eye search's 5.5x fewer vision tokens don't make up for ~2 calls and ~35 output tokens per search (vs 1 call and ~15 tokens); the gap narrows under load (0.64x at c=1, 0.90x at c=16 for BF16).
- **Measurement trap (fixed before reporting):** the first end-to-end run showed eye search *ahead* under load (R1-ft capped at 7.3 q/s). The cause was the client: PNG-encoding a full 640x480 frame costs ~90 ms of CPU and ran inside the asyncio event loop, serializing all concurrent full-frame searches (thumbnails cost 1.3 ms). Encoding now runs in a worker thread (`VLMClient.aencode`; PIL releases the GIL, 6.3x speed-up with 8 threads), and the end-to-end numbers were re-measured. Accuracy results are unaffected.
- Per call at concurrency 16 (req/s, BF16 -> FP8): thumbnail 36.0 -> 41.1, crop 28.8 -> 33.6, full frame 14.4 -> 16.8.
- **CUDA graphs vs `--enforce-eager`** (FP8): eager is 3.2-3.5x slower per call at concurrency 1 (thumbnail 789 vs 224 ms, full frame 825 vs 261 ms) and 2.7x lower thumbnail throughput at concurrency 8 (8.7 vs 23.6 req/s). Kernel-launch overhead dominates a 2B model's decode step. Eager starts faster (48 s vs ~150 s, no compile).

**W4A16: skipped.** It needs a calibration pass (GPTQ) to be trustworthy, and the L4 has no native INT4 matmul (Marlin kernels dequantize). FP8 already showed no measurable accuracy cost, with the decode speed-up limited by the per-call overhead above, so it did not fit the ~15 min stretch budget.

**Headline.** Fine-tuning made eye search significantly better (+6.4 pp accuracy, +7.8 pp found, and false positives at about 1%), and FP8 made it 24% faster at no measurable cost. But on COCO-sized frames the same fine-tuning made plain full-frame inference better still: full frame is more accurate (94.2 vs 85.6%) and serves more searches per L4. Eye search's real advantage is the pixel and vision-token budget (18% of the pixels, 49 vs 272 vision tokens), which matters for bandwidth, and for larger frames where a full frame costs far more tokens. It doesn't win on accuracy or throughput for 640x480 frames on a datacenter GPU.

### Step 8b: frame-size scaling and the look-again mask

**High-resolution benchmark from the frozen eval set** (`foveated_search/data/mosaic.py`; `make mosaics`). The 308 test queries whose image fits a 640x480 cell (images are never resized; the other 142 test queries are portrait or taller and are excluded at every size) become 2x2 (1280x960) and 3x3 (1920x1440) mosaics. The query image goes in a seeded random cell, centred on gray (128); the other cells hold manifest images with no category of the query's supercategory (crowd included; for "food" also none of the 13 food-adjacent classes). Boxes are shifted into mosaic coordinates. Manifests: `results/step8b_mosaic_manifest_2x2.jsonl` (sha256 `71943fd6591f7572561dd6cbd7811184b0d7be72d1e093bed045db0c40162ec5`) and `_3x3.jsonl` (`6006ced82bb0f93fb5cdc079f61f17ec374672bb2ccc6f84dfe356fbf33a3db1`). Regeneration is identical (test). Rendered frames are on the local disk only (2.1 GB, 32 s). Contact sheet: `results/step8b_mosaic_examples.png`. 1x1 numbers below are on the same 308 queries.

**Server:** fine-tuned FP8, prefix caching off, plus `configs/serve_hires.yaml` (image_max_pixels 2,764,800 = exactly 1920x1440, max_model_len 6,144). Verified: the full frame reaches the model at native resolution at every size, and processor tokens == server tokens: 300 / 1,200 / 2,700 (`results/step8b_token_check.json`). 0 parse failures in all 15 accuracy runs.

| frame | strategy | accuracy % [95% CI] | FP % | vision tokens | pixels % | time-to-answer p50 (ms, conc 1) | searches/s @16 |
|---|---|---|---|---|---|---|---|
| 1x1 | full frame | 93.5 [90.6, 96.1] | 4.3 | 261 | 100 | 257 | 17.4 |
| 1x1 | eye search 288 px | **93.5** [90.9, 96.1] | 1.4 | **98** | 39.0 | 515 | 15.8 |
| 1x1 | eye search 96 px | 87.0 | 1.4 | 48 | 19.1 | 507 | 15.8 |
| 2x2 | full frame | **89.3** [85.7, 92.9] | 7.1 | 1,200 | 100 | 498 | 4.2 |
| 2x2 | eye search 288 px | 85.4 [81.5, 89.6] | 1.4 | **110** | 9.2 | 515 | **13.3** |
| 2x2 | downscale (eye-96 budget) | 74.4 | 5.7 | 54 | 4.5 | 233 | 22.2 |
| 3x3 | full frame | **88.0** [84.4, 91.6] | 10.0 | 2,700 | 100 | 1,048 | 1.7 |
| 3x3 | eye search 288 px | 73.4 [68.5, 78.2] | 4.3 | **129** | 4.8 | 515 | **10.0** |
| 3x3 | eye search 192 px | 59.1 | 5.7 | 88 | 3.2 | 512 | 10.1 |
| 3x3 | eye search 96 px | 39.9 | 2.9 | 78 | 2.9 | 795 | 8.8 |
| 3x3 | downscale (eye-96 budget) | 60.1 | 25.7 | 70 | 2.6 | 234 | 14.1 |

(All 15 rows with CIs, latency p95 and accuracy per 1k vision tokens: `results/step8b_scaling.csv`, `results/table.md`, figure `results/step8b_scaling.png`. Downscale budgets = eye-96 mean pixels: 49,261 / 62,002 / 79,702 px.)

**Findings:**
- **Throughput crossover between 1x1 and 2x2.** Full-frame cost grows with the frame: prefill of 1,200 / 2,700 vision tokens drops it from 17.4 searches/s (1x1) to 4.2 (2x2) and 1.7 (3x3). Eye search's cost is nearly flat (98 -> 129 vision tokens for 288 px), so it serves **3.2x** (2x2) and **5.9x** (3x3) more searches per L4 at concurrency 16. At 1x1 they are roughly even (15.8-16.9 vs 17.4).
- **Single-search latency** follows the same pattern: at concurrency 1, full frame goes 257 -> 498 -> 1,048 ms while eye search 288 px stays at ~515 ms. They cross at 2x2.
- **Accuracy: no crossover.** Full frame stays the most accurate at every size (93.5 / 89.3 / 88.0%). Eye search 288 px ties it at 1x1 (93.5% with 37% of the vision tokens), trails by 3.9 pp at 2x2 (CIs overlap) and by 14.6 pp at 3x3 (CIs don't overlap). The eye-search loss at large frames is the glance: a fixed-size thumbnail of a 3x3 mosaic gives each 640x480 cell only ~32 / 64 / 96 px, and accuracy falls with it (96 px: 39.9%, 192 px: 59.1%, 288 px: 73.4%). The confirm step still keeps false positives low (1.4-5.7% vs 10% for full frame at 3x3).
- **Accuracy per vision token:** eye search is 2.7x (1x1), 10x (2x2) and 17x (3x3) more accuracy-efficient than full frame (9.6 / 7.8 / 5.7 vs 3.6 / 0.74 / 0.33 accuracy points per 1k vision tokens, 288 px).
- **Downscale at eye search's pixel budget** is the fastest strategy but degrades worst on big frames (3x3: 60.1% accuracy, 25.7% FP): a uniformly shrunk 3x3 mosaic leaves objects a few pixels wide.
- **Where eye search wins:** large frames where throughput or the token / bandwidth budget matters: at 1280x960 it keeps 96% of full frame's accuracy (85.4 vs 89.3) at 3.2x the throughput and 9% of the pixels. Where accuracy is all that matters, full frame (at native resolution) is still best at every size we tested.

**Measurement trap (fixed):** the runner only switched to latency mode (concurrency 1) for a split named exactly `latency`, so the first `latency_fit` / `latency_m2` / `latency_m3` runs ran at concurrency 8 (61 searches in 5 s gave it away). Fixed with `run_mode()` (any `latency*` split; unit-tested); all 15 latency runs were deleted and re-run at concurrency 1.

**Look-again mask (inference-only fix, 1x1 test, 450 queries, FP8, same server):** on fixations 2+, the checked crop regions are filled gray in the thumbnail, in addition to listing the points in text (`search.look_again_mask`).

| | mask off (same server) | mask on |
|---|---|---|
| found % | 81.4 | **86.9** (+5.6 pp [+2.8, +8.3]) |
| accuracy % | 84.9 | **89.3** (+4.4 pp [+2.2, +6.7]); 24 wrong->right, 4 right->wrong |
| FP % | 1.1 | 1.1 |
| repeats | 78 | **21** |
| searches with a rejected crop -> took a 2nd fixation | 53 -> 19 | 53 -> **44** |
| found after the 2nd fixation | 7 | **27** |
| calls / vision tokens per search | 2.07 / 49 | 2.11 / 53 |

Against R6: +4.2 pp accuracy [+2.2, +6.4]. The mask fixes most of the "look again" problem from Step 8 without retraining, for 4 more vision tokens per search. It breaks thumbnail byte-identity, which costs only the 5.6% prefix-cache hit rate measured in Step 8. **Kept** as `search.look_again_mask`. The default stays `false` so R1-R7 remain reproducible from config; recommended `true` for eye search from Step 9 on.

### Step 9: photo replay demo

**Pipeline** (`foveated_search/demo/`): `intake.py` (photo -> clean copy) -> `run_demo.py` (traces) -> `replay.py` + `web/template.html` (one offline HTML page). Server: `make serve-demo` (fine-tuned FP8 from the private Hub repo, native resolution up to 1920x1440, prefix + mm caches off). `run_demo` reads `/v1/models` and refuses to run unless the server is serving `honey613/qwen3-vl-2b-foveated-search-fp8`.

**Intake, measured:**
- pillow-heif 1.8.1 (libheif 1.23.6) applies the HEIF rotation when it opens the file, sets the EXIF orientation tag to 1 and keeps the old value in `info["original_orientation"]`. So `ImageOps.exif_transpose` afterwards is a no-op, and a HEIC is rotated exactly once (tested: orientation 6 -> portrait, marker in the right corner, same result as the JPEG path).
- Stripping is done by rebuilding the image from raw pixels (`Image.frombytes`), not by deleting tags. The output check reads the decoded EXIF (must have 0 tags, no GPS IFD), the `info` dict (no exif / icc / xmp / comment), and walks the JPEG segment headers: only APP0 (JFIF) may come before the image data. A plain byte search for `b"GPS"` was rejected: 3 bytes match random JPEG data by chance about 4% of the time on a 600 KB file.
- An embedded ICC profile (phones: Display P3) is converted to sRGB before stripping, so colours stay right without the profile.
- The real processor at the demo server's limits keeps the clean copy at native size: a 4032x3024 photo -> 1920x1440 = 2,700 vision tokens (exactly `image_max_pixels`); 1080 px sides round up to 1088 (grid of 32), which is not a downscale. Intake fails if the processor would shrink the image by more than one grid step.

**Timing protocol:** per image, 1 unrecorded warmup round of both strategies, then full frame and eye search once each, sequentially, at concurrency 1. Time to answer = sum of measured call times (client on the GPU host, localhost). The replay page plays the calls on that same timeline (call k starts when call k-1 ends).

**Replay integrity:** `make replay` re-creates every image the model was sent (thumbnail, masked thumbnails on fixations 2+, crops, full frame) from the clean photo with the search code and checks its sha256 against the trace; any difference fails the build. The page embeds those exact PNGs for eye search; the full frame is embedded as a downsized JPEG for display only.

**Pipeline test before the user's photo** (Step 8b mosaic frames, demo settings, both strategies correct on all three; runs deleted afterwards, copy of the page at `/content/fvs_local/demo_test/replay_test.html`, local disk only):

| frame | query | full frame | eye search |
|---|---|---|---|
| 2x2 q0005 (1280x960) | food | found 0.50 s, 1,200 tok, 2.1 MB | found 0.57 s (fixation 1, "cake"), 112 tok, 226 KB, 9.3% px |
| 3x3 q0313 (1920x1440) | truck | found 1.09 s, 2,700 tok, 4.9 MB | found 1.28 s (fixation 3, 2 crops rejected, mask on), 336 tok, 706 KB, 12.4% px |
| 3x3 q0021 (1920x1440) | food (negative) | absent 0.98 s | absent 0.14 s (glance only), 63 tok, 126 KB, 2.3% px |

The first 3x3 candidate (umbrella, q0171) took 2 fixations in Step 8b but only 1 with the demo settings, so the truck query was used to exercise the look-again path. In headless Chromium (Playwright, scratch venv on the local disk) the page loaded from `file://` with 0 console errors and 0 network requests at 1440 px and 390 px width (no horizontal scroll), and each replay took as long as the measured time to answer (1.27 s wall for 1.28 s measured).

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
- transformers 5.17: `TrainingArguments` has no `warmup_ratio` (use a float `warmup_steps`), and Qwen3-VL training needs the processor's `mm_token_type_ids` (multimodal RoPE). See Step 7.
- Throughput benchmarks: CPU work in the asyncio event loop (PNG-encoding a 640x480 frame, ~90 ms) silently caps concurrency and can flip a comparison. Keep encoding off the loop and sanity-check end-to-end throughput against per-call throughput.
- Prefix-cache measurements: a run that repeats queries the same server just served measures a hot cache (51% hits vs 5% cold). Restart the server, or use disjoint queries, before measuring.
- Split-name conventions are code: the runner chose latency mode only for the exact split name `latency`, so new `latency_*` splits silently ran at concurrency 8. Mode detection is now `run_mode()` (any `latency*`), with a unit test.
- pillow-heif rotates HEIC on open and resets the orientation tag; don't add a second manual rotation from `original_orientation` (Step 9).
