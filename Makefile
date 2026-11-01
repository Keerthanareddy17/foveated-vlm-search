SHELL := /bin/bash
.DEFAULT_GOAL := help
SERVE_PY := .venv-serve/bin/python
TRAIN_PY := .venv-train/bin/python

.PHONY: help setup check storage gpu test clean serve serve-stop serve-status bench-latency eval-manifest baselines gen-data train quantize demo

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

eval-manifest: ## Build frozen eval manifest (Step 3)
	@echo "not implemented yet (Step 3)"

baselines: ## Run R1-R4 baselines + thumbnail sweep (Step 5)
	@echo "not implemented yet (Step 5)"

gen-data: ## Generate training data (Step 6)
	@echo "not implemented yet (Step 6)"

train: ## LoRA fine-tune + merge (Step 7)
	@echo "not implemented yet (Step 7)"

quantize: ## FP8 quantization (Step 8)
	@echo "not implemented yet (Step 8)"

demo: ## Live webcam demo server (Step 9)
	@echo "not implemented yet (Step 9)"
