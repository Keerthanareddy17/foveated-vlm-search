"""Build the `vllm serve` command line from the config (no flags hard-coded in shell scripts).

Usage (from serve.sh):  python -m foveated_search.serving.launch args   -> NUL-separated argv
                        python -m foveated_search.serving.launch url    -> http://host:port
"""

from __future__ import annotations

import json
import sys

from foveated_search.config import Config, load_config


def mm_processor_kwargs(cfg: Config) -> dict | None:
    """Explicit image resize bounds for the Qwen processor (same dict the client-side processor uses)."""
    lo, hi = cfg.get_key("server.image_min_pixels"), cfg.get_key("server.image_max_pixels")
    if lo is None and hi is None:
        return None
    if lo is None or hi is None:
        raise ValueError("Set both server.image_min_pixels and server.image_max_pixels, or neither.")
    return {"size": {"shortest_edge": int(lo), "longest_edge": int(hi)}}


def vllm_args(cfg: Config) -> list[str]:
    s = cfg["server"]
    args = [
        "serve", str(cfg.get_key("model.id")),
        "--served-model-name", str(cfg.get_key("model.served_name")),
        "--host", str(s["host"]),
        "--port", str(s["port"]),
        "--dtype", str(s["dtype"]),
        "--max-model-len", str(s["max_model_len"]),
        "--gpu-memory-utilization", str(s["gpu_memory_utilization"]),
        "--max-num-seqs", str(s["max_num_seqs"]),
        "--limit-mm-per-prompt", json.dumps({"image": int(s["max_images_per_prompt"]), "video": 0}),
        "--mm-processor-cache-gb", str(s["mm_processor_cache_gb"]),
        "--seed", str(cfg["seed"]),
        # Use our sampling params, not the model's generation_config.json (temperature 0.7 etc.).
        "--generation-config", "vllm",
    ]
    args.append("--enable-prefix-caching" if s["enable_prefix_caching"] else "--no-enable-prefix-caching")
    if s["enforce_eager"]:
        args.append("--enforce-eager")
    if s["quantization"]:
        args += ["--quantization", str(s["quantization"])]
    mmk = mm_processor_kwargs(cfg)
    if mmk:
        args += ["--mm-processor-kwargs", json.dumps(mmk)]
    return args


def server_env(cfg: Config) -> dict[str, str]:
    """Environment variables the vLLM process needs (also config-driven)."""
    return {"VLLM_USE_FLASHINFER_SAMPLER": "1" if cfg.get_key("server.flashinfer_sampler") else "0"}


def base_url(cfg: Config) -> str:
    return f"http://{cfg.get_key('server.host')}:{cfg.get_key('server.port')}"


if __name__ == "__main__":
    cfg = load_config(set_seed=False)
    what = sys.argv[1] if len(sys.argv) > 1 else "args"
    if what == "args":
        sys.stdout.write("\0".join(vllm_args(cfg)))
    elif what == "env":
        for k, v in server_env(cfg).items():
            print(f"export {k}={v}")
    elif what == "url":
        print(base_url(cfg))
    else:
        raise SystemExit(f"unknown: {what}")
