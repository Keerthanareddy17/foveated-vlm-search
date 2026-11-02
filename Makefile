SHELL := /bin/bash
.DEFAULT_GOAL := help
SERVE_PY := .venv-serve/bin/python
TRAIN_PY := .venv-train/bin/python

.PHONY: help setup check storage serve-demo demo-smoke gpu eval table mosaics scaling gen-data base-on-val train-bench train merge push-hub test clean serve serve-stop serve-status bench-latency step4-check eval-manifest baselines gen-data train quantize push-hub-fp8 demo intake demo-batch replay demo-sweep readme-tables figures pages record-demo

help: ## List all targets
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup: ## Create/refresh venvs, env file, Drive folders (idempotent, every session)
	bash scripts/setup_colab.sh

check: ## Verify GPU, venvs, Drive, HF token, disk; writes results/env_report.json
	python3 scripts/check_env.py

storage: ## Project Drive usage vs 500 MB budget, local cache usage, free disk (read-only)
	@bash scripts/storage_report.sh

gpu: ## Show nvidia-smi
	nvidia-smi

test: ## Run unit tests (pytest in .venv-serve, GPU-free)
	$(SERVE_PY) -m pytest -q

clean: ## Remove caches inside the repo (never touches Drive or venvs)
	find . -path ./.venv-serve -prune -o -path ./.venv-train -prune -o -path ./.venv-quant -prune -o \
	  \( -name __pycache__ -o -name .pytest_cache -o -name '*.egg-info' -o -name .ipynb_checkpoints \) -print -exec rm -rf {} +

serve: ## Start vLLM in background from configs (FVS_CONFIG_OVERLAYS=... for variants), wait for /health
	bash scripts/serve.sh start

DEMO_OVERLAYS := configs/serve_demo.yaml,configs/serve_hires.yaml,configs/demo.yaml

serve-demo: ## Demo server: fine-tuned FP8 from the private HF repo, native-resolution frames, demo search settings
	FVS_CONFIG_OVERLAYS=$(DEMO_OVERLAYS) bash scripts/serve.sh start

demo-smoke: ## One eye-search query against the demo server with the demo settings (make serve-demo first)
	source scripts/env.sh && FVS_CONFIG_OVERLAYS=$(DEMO_OVERLAYS) $(SERVE_PY) scripts/demo_smoke.py

serve-stop: ## Stop vLLM and confirm GPU memory is freed
	bash scripts/serve.sh stop

serve-status: ## Show vLLM pid, /health, served model, GPU memory
	bash scripts/serve.sh status

bench-latency: ## Per-call latency: thumbnail vs crop vs full frame (OUT=results/step2_latency.csv)
	source scripts/env.sh && $(SERVE_PY) scripts/bench_latency.py --out $(or $(OUT),results/step2_latency.csv)

eval-manifest: ## Step 3 end to end: COCO annotations -> manifest + dev subset -> fetch/verify images -> stats + contact sheet
	source scripts/env.sh && $(SERVE_PY) -m foveated_search.data.manifest all

step4-check: ## Step 4 live check: 4 strategies on 5 dev queries -> traces, summary, trace images (server must be up)
	source scripts/env.sh && $(SERVE_PY) scripts/step4_check.py

eval: ## One eval run: make eval RUN=<name> [STRATEGY=<s> SPLIT=dev|test|all|latency SET="a.b=1,c.d=x"] (resumable)
	source scripts/env.sh && $(SERVE_PY) -m foveated_search.eval.runner --run $(RUN) \
	  $(if $(STRATEGY),--strategy $(STRATEGY)) $(if $(SPLIT),--split $(SPLIT)) $(if $(SET),--set "$(SET)")

baselines: ## All R1-R4 runs from configs/runs.yaml (accuracy on all 500 + concurrency-1 latency subset), then the table
	source scripts/env.sh && for r in $$($(SERVE_PY) -c "import yaml;print(' '.join(yaml.safe_load(open('configs/runs.yaml'))['baselines']))"); do \
	  $(SERVE_PY) -m foveated_search.eval.runner --run $$r || exit 1; done
	$(MAKE) table

table: ## Regenerate results/table.md from results/runs/ (never hand-edit the table)
	source scripts/env.sh && $(SERVE_PY) -m foveated_search.eval.table

gen-data: ## Step 6 end to end: plan -> fetch train2017 subset -> generate -> stats + grid -> processor check (.venv-train)
	source scripts/env.sh && $(SERVE_PY) -m foveated_search.data.train_data all
	source scripts/env.sh && $(TRAIN_PY) -m foveated_search.train.check_data
	source scripts/env.sh && $(SERVE_PY) -m foveated_search.data.train_data report

mosaics: ## Step 8b: 2x2 / 3x3 mosaic manifests from the frozen eval set + rendered frames (local) + contact sheet
	source scripts/env.sh && $(SERVE_PY) -m foveated_search.data.mosaic

scaling: ## Step 8b: downscale budgets, throughput @16, scaling CSV + figure, table (server must be up for budgets/throughput)
	source scripts/env.sh && $(SERVE_PY) -m foveated_search.eval.scaling report && $(SERVE_PY) scripts/step8b_figure.py && $(MAKE) table

base-on-val: ## Step 6: score the base model on val samples per type (server must be up)
	source scripts/env.sh && $(SERVE_PY) scripts/step6_base_on_val.py

train-bench: ## Step 7: short training benchmark (BENCH=40 steps): speed + memory, then cleaned up. vLLM must be stopped
	@curl -sf http://127.0.0.1:8000/health >/dev/null && { echo "vLLM is running: make serve-stop first"; exit 1; } || true
	source scripts/env.sh && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $(TRAIN_PY) -m foveated_search.train.train_lora --bench $(or $(BENCH),40)

train: ## Step 7: full LoRA run in the background (nohup; resumes from the latest checkpoint). vLLM must be stopped
	@curl -sf http://127.0.0.1:8000/health >/dev/null && { echo "vLLM is running: make serve-stop first"; exit 1; } || true
	@mkdir -p logs; source scripts/env.sh && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup $(TRAIN_PY) -m foveated_search.train.train_lora > logs/train.log 2>&1 & \
	  echo $$! > logs/train.pid; echo "command: $(TRAIN_PY) -m foveated_search.train.train_lora"; \
	  echo "log: logs/train.log   pid: $$(cat logs/train.pid)"

merge: ## Step 7: merge the LoRA adapter into the BF16 base, save locally, diff configs vs base
	source scripts/env.sh && $(TRAIN_PY) -m foveated_search.train.merge

push-hub: ## Step 7: push the merged model to a PRIVATE HF repo (checks write access first)
	source scripts/env.sh && $(TRAIN_PY) -m foveated_search.train.push_hub

QUANT_PY := .venv-quant/bin/python
FT_MODEL := qwen3-vl-2b-foveated-search

quantize: ## Step 8: FP8_DYNAMIC (decoder only) of the merged fine-tuned model with llm-compressor (.venv-quant)
	source scripts/env.sh && $(QUANT_PY) -m foveated_search.quant.fp8_dynamic --model $$FVS_MODELS_DIR/$(FT_MODEL) \
	  --out $$FVS_MODELS_DIR/$(FT_MODEL)-fp8 --report results/step8_fp8_quantize.json

push-hub-fp8: ## Step 8: push the FP8 model to a PRIVATE HF repo (<user>/<hub_repo_name>-fp8)
	source scripts/env.sh && $(TRAIN_PY) -m foveated_search.train.push_hub --kind fp8 \
	  --model-dir $$FVS_MODELS_DIR/$(FT_MODEL)-fp8 --repo-name $(FT_MODEL)-fp8

intake: ## Step 9: phone photos in test_pictures/ -> upright, metadata-free, 1920 px copies in test_pictures/clean/
	source scripts/env.sh && FVS_CONFIG_OVERLAYS=$(DEMO_OVERLAYS) $(SERVE_PY) -m foveated_search.demo.intake

demo: ## Step 9: make demo IMAGE=test_pictures/clean/x.jpg QUERY="food" -> results/demo/ (make serve-demo first)
	@test -n "$(IMAGE)" -a -n "$(QUERY)" || { echo 'usage: make demo IMAGE=test_pictures/clean/<photo>.jpg QUERY="food"'; exit 1; }
	source scripts/env.sh && FVS_CONFIG_OVERLAYS=$(DEMO_OVERLAYS) $(SERVE_PY) -m foveated_search.demo.run_demo --image "$(IMAGE)" --query "$(QUERY)"

demo-batch: ## Step 9: every (image, query) pair in configs/demo_queries.yaml [ONLY=positive2] (make serve-demo first)
	source scripts/env.sh && FVS_CONFIG_OVERLAYS=$(DEMO_OVERLAYS) $(SERVE_PY) -m foveated_search.demo.run_demo --batch $(if $(ONLY),--only "$(ONLY)")

demo-sweep: ## Step 9: eye-search thumbnail sweep on the demo photos -> results/demo/thumbnail_sweep.csv (make serve-demo first)
	source scripts/env.sh && FVS_CONFIG_OVERLAYS=$(DEMO_OVERLAYS) $(SERVE_PY) scripts/demo_thumbnail_sweep.py

readme-tables: ## Step 10: README.md from scripts/readme_template.md + numbers computed from results/ (CHECK=1 lists untraced digits)
	source scripts/env.sh && $(SERVE_PY) scripts/build_readme_tables.py $(if $(CHECK),--check)

figures: ## Step 10: README figures (light + dark) -> docs/assets/
	source scripts/env.sh && $(SERVE_PY) scripts/make_figures.py

pages: ## Step 10: GitHub Pages copy of the replay page -> docs/index.html
	source scripts/env.sh && FVS_CONFIG_OVERLAYS=$(DEMO_OVERLAYS) $(SERVE_PY) -m foveated_search.demo.replay --pages

record-demo: ## Step 10: record the replay page -> docs/assets/demo.gif + demo.mp4 (Playwright + ffmpeg)
	bash scripts/record_demo.sh

replay: ## Step 9: build the self-contained replay page results/demo/replay.html from results/demo/*.json
	source scripts/env.sh && FVS_CONFIG_OVERLAYS=$(DEMO_OVERLAYS) $(SERVE_PY) -m foveated_search.demo.replay
