#!/usr/bin/env python3
"""Environment check: PASS/FAIL table + results/env_report.json. Exit 1 on any FAIL.

Stdlib only (runs under any python). Each venv is probed with its own interpreter.
Never reads the HF token contents; only checks that the file exists.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DRIVE_ROOT = Path("/content/drive/MyDrive")
PROJECT_DRIVE_DIR = Path(os.environ.get("PROJECT_DRIVE_DIR", DRIVE_ROOT / "foveated-vlm-search"))
HF_TOKEN = Path.home() / ".cache" / "huggingface" / "token"
MIN_FREE_GB = 30

VENVS = {
    "serve": {
        "path": REPO / ".venv-serve",
        "modules": ["vllm", "torch", "transformers", "openai", "fastapi", "PIL", "pycocotools"],
        "extra": [],
    },
    "train": {
        "path": REPO / ".venv-train",
        "modules": ["torch", "transformers", "peft", "trl", "datasets", "accelerate"],
        "extra": ["transformers.Qwen3VLForConditionalGeneration"],
    },
    # llmcompressor lives here: its compressed-tensors pin conflicts with vLLM's (NOTES.md).
    "quant": {
        "path": REPO / ".venv-quant",
        "modules": ["llmcompressor", "compressed_tensors", "torch", "transformers"],
        "extra": [],
    },
}

# Runs inside each venv. Prints one JSON line.
PROBE = r"""
import importlib, json, sys
mods, extra = json.loads(sys.argv[1]), json.loads(sys.argv[2])
out = {"python": sys.version.split()[0], "modules": {}, "extra": {}, "cuda": None}
for m in mods:
    try:
        mod = importlib.import_module(m)
        ver = getattr(mod, "__version__", None)
        if ver is None:
            try:
                from importlib.metadata import version as _v
                ver = _v(m)
            except Exception:
                ver = "?"
        out["modules"][m] = {"ok": True, "version": ver}
    except Exception as e:
        out["modules"][m] = {"ok": False, "error": f"{type(e).__name__}: {e}"[:200]}
for x in extra:
    mod, attr = x.rsplit(".", 1)
    try:
        getattr(importlib.import_module(mod), attr)
        out["extra"][x] = {"ok": True}
    except Exception as e:
        out["extra"][x] = {"ok": False, "error": f"{type(e).__name__}: {e}"[:200]}
try:
    import torch
    out["cuda"] = {"available": torch.cuda.is_available(), "torch_cuda": torch.version.cuda,
                   "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
except Exception as e:
    out["cuda"] = {"available": False, "error": str(e)[:200]}
print("__PROBE__" + json.dumps(out))
"""

rows: list[tuple[str, bool, str]] = []


def add(name: str, ok: bool, detail: str = "") -> None:
    rows.append((name, ok, detail))


def check_gpu() -> dict:
    try:
        q = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,compute_cap,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, check=True, timeout=30,
        ).stdout.strip().splitlines()[0]
        name, mem, cc, drv = [s.strip() for s in q.split(",")]
        gpu = {"name": name, "memory_total": mem, "compute_cap": cc, "driver": drv}
    except Exception as e:
        add("GPU is L4, cc 8.9", False, f"nvidia-smi failed: {e}")
        return {}
    add("GPU is L4, cc 8.9", "L4" in name and cc == "8.9", f"{name}, cc {cc}, {mem}, driver {drv}")
    return gpu


def check_venv(label: str, spec: dict) -> dict:
    py = spec["path"] / "bin" / "python"
    if not py.exists():
        add(f"[{label}] venv exists", False, f"{py} missing (run make setup)")
        return {}
    r = subprocess.run(
        [str(py), "-c", PROBE, json.dumps(spec["modules"]), json.dumps(spec["extra"])],
        capture_output=True, text=True, timeout=600,
    )
    line = next((l for l in r.stdout.splitlines() if l.startswith("__PROBE__")), None)
    if line is None:
        add(f"[{label}] probe ran", False, (r.stderr or r.stdout)[-300:])
        return {}
    res = json.loads(line[len("__PROBE__"):])
    c = res["cuda"] or {}
    add(f"[{label}] torch CUDA available", bool(c.get("available")),
        f"cuda {c.get('torch_cuda')}, {c.get('device')}" if c.get("available") else c.get("error", "not available"))
    for m, v in res["modules"].items():
        add(f"[{label}] import {m}", v["ok"], v.get("version") if v["ok"] else v["error"])
    for x, v in res["extra"].items():
        add(f"[{label}] {x}", v["ok"], "importable" if v["ok"] else v["error"])
    return res


def check_drive() -> dict:
    mounted = DRIVE_ROOT.is_dir()
    add("Drive mounted", mounted, str(DRIVE_ROOT) if mounted else "not mounted: run the notebook setup cell")
    writable = False
    if mounted:
        try:
            PROJECT_DRIVE_DIR.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=PROJECT_DRIVE_DIR, prefix=".write_test_", delete=True) as f:
                f.write(b"ok")
                f.flush()
            writable = True
        except Exception as e:
            add("Drive project dir writable", False, str(e)[:200])
            return {"mounted": mounted, "writable": False}
    if mounted:
        add("Drive project dir writable", writable, str(PROJECT_DRIVE_DIR))
    return {"mounted": mounted, "writable": writable, "project_dir": str(PROJECT_DRIVE_DIR)}


def main() -> int:
    t0 = dt.datetime.now(dt.timezone.utc)
    report: dict = {"timestamp_utc": t0.isoformat(timespec="seconds")}
    report["gpu"] = check_gpu()
    report["venvs"] = {k: check_venv(k, v) for k, v in VENVS.items()}
    report["drive"] = check_drive()
    tok = HF_TOKEN.is_file()  # existence only; contents never read
    add("HF token file present", tok, "~/.cache/huggingface/token" if tok else "missing: log in from the notebook")
    report["hf_token_present"] = tok
    report["hf_home_set"] = "HF_HOME" in os.environ
    free_gb = shutil.disk_usage("/content").free / 1e9
    add(f"Free disk on /content >= {MIN_FREE_GB} GB", free_gb >= MIN_FREE_GB, f"{free_gb:.1f} GB free")
    report["disk_free_gb_content"] = round(free_gb, 1)

    w = max(len(r[0]) for r in rows)
    print(f"\n{'CHECK'.ljust(w)}  RESULT  DETAIL")
    print("-" * (w + 60))
    for name, ok, detail in rows:
        print(f"{name.ljust(w)}  {'PASS' if ok else 'FAIL'}    {detail}")
    n_fail = sum(not ok for _, ok, _ in rows)
    print("-" * (w + 60))
    print(f"{len(rows) - n_fail}/{len(rows)} passed" + (f", {n_fail} FAILED" if n_fail else ""))

    report["checks"] = [{"name": n, "pass": ok, "detail": d} for n, ok, d in rows]
    report["all_pass"] = n_fail == 0
    report["duration_s"] = round((dt.datetime.now(dt.timezone.utc) - t0).total_seconds(), 1)
    out = REPO / "results" / "env_report.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {out.relative_to(REPO)}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
