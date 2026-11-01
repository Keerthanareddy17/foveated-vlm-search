SHELL := /bin/bash
.DEFAULT_GOAL := help
SERVE_PY := .venv-serve/bin/python
TRAIN_PY := .venv-train/bin/python

.PHONY: help setup check storage gpu eval table test clean serve serve-stop serve-status bench-latency step4-check eval-manifest baselines gen-data train quantize demo

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

gen-data: ## Generate training data (Step 6)
	@echo "not implemented yet (Step 6)"

train: ## LoRA fine-tune + merge (Step 7)
	@echo "not implemented yet (Step 7)"

quantize: ## FP8 quantization (Step 8)
	@echo "not implemented yet (Step 8)"

demo: ## Live webcam demo server (Step 9)
	@echo "not implemented yet (Step 9)"
