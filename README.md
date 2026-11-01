# foveated-vlm-search

Find things in live camera frames while barely looking: a small VLM (Qwen3-VL-2B) glances at a tiny thumbnail, then fixates on a few sharp crops until it confirms the target.

**Status: work in progress.** The full write-up (results table, demo, how it works) comes at the end of the project.

## Quick start (Google Colab, L4 GPU)

1. In the notebook: mount Google Drive at `/content/drive` and log in to Hugging Face.
2. In a terminal, from the repo root:

```bash
make setup   # creates .venv-serve and .venv-train with uv (idempotent; re-run every session)
make check   # PASS/FAIL table for GPU, venvs, Drive, HF token, disk -> results/env_report.json
make test    # unit tests
make help    # all targets
```
