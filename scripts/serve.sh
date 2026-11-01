#!/usr/bin/env bash
# vLLM server control. All server arguments come from configs/ (see foveated_search/serving/launch.py).
#   scripts/serve.sh start|stop|status
# Variants via overlays:  FVS_CONFIG_OVERLAYS=configs/serve_fp8_online.yaml scripts/serve.sh start
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
[ -f scripts/env.sh ] || { echo "ERROR: scripts/env.sh missing, run 'make setup' first." >&2; exit 1; }
source scripts/env.sh
PY="$VENV_SERVE/bin/python"
LOG_DIR="$REPO/logs"; mkdir -p "$LOG_DIR"
PID_FILE="$LOG_DIR/vllm.pid"
LOG_FILE="$LOG_DIR/vllm.log"

is_running() { [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; }
gpu_mem() { nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader; }

start() {
  if is_running; then echo "vLLM already running (pid $(cat "$PID_FILE")). Use 'make serve-stop' first."; exit 1; fi
  local url; url=$("$PY" -m foveated_search.serving.launch url)
  if curl -sf "$url/health" >/dev/null 2>&1; then echo "ERROR: something already answers on $url (not ours)." >&2; exit 1; fi
  mapfile -d '' ARGS < <("$PY" -m foveated_search.serving.launch args)
  local timeout; timeout=$("$PY" -c "from foveated_search.config import load_config as l; print(l(set_seed=False).get_key('server.health_timeout_s'))")
  eval "$("$PY" -m foveated_search.serving.launch env)"
  # vLLM/FlashInfer JIT helpers (ninja, etc.) live in the venv's bin; we don't activate the venv.
  export PATH="$VENV_SERVE/bin:$PATH"
  echo "==> env: VLLM_USE_FLASHINFER_SAMPLER=$VLLM_USE_FLASHINFER_SAMPLER"
  echo "==> overlays: ${FVS_CONFIG_OVERLAYS:-none}"
  echo "==> vllm ${ARGS[*]}"
  # Rotate the previous log so each start has a clean one.
  [ -f "$LOG_FILE" ] && mv "$LOG_FILE" "$LOG_DIR/vllm.$(date +%Y%m%d-%H%M%S).log"
  local t0; t0=$(date +%s)
  nohup "$VENV_SERVE/bin/vllm" "${ARGS[@]}" > "$LOG_FILE" 2>&1 &
  echo $! > "$PID_FILE"
  echo "==> started pid $(cat "$PID_FILE"), log: logs/vllm.log; waiting for $url/health (timeout ${timeout}s)"
  while true; do
    if curl -sf "$url/health" >/dev/null 2>&1; then
      local dt=$(( $(date +%s) - t0 ))
      echo "==> READY in ${dt}s at $url"
      echo "    GPU memory used: $(gpu_mem)"
      echo "startup_seconds=$dt" > "$LOG_DIR/vllm.startup"
      return 0
    fi
    if ! is_running; then
      echo "==> FAILED: server exited during startup. Last log lines:" >&2; tail -n 40 "$LOG_FILE" >&2; rm -f "$PID_FILE"; exit 1
    fi
    if (( $(date +%s) - t0 > timeout )); then
      echo "==> FAILED: no /health after ${timeout}s. Last log lines:" >&2; tail -n 40 "$LOG_FILE" >&2; stop; exit 1
    fi
    sleep 3
  done
}

stop() {
  if ! is_running; then echo "vLLM not running."; rm -f "$PID_FILE"; else
    local pid; pid=$(cat "$PID_FILE")
    echo "==> stopping vLLM pid $pid"
    kill -TERM "$pid" 2>/dev/null || true
    for _ in $(seq 1 40); do kill -0 "$pid" 2>/dev/null || break; sleep 0.5; done
    if kill -0 "$pid" 2>/dev/null; then echo "    still alive, SIGKILL"; pkill -KILL -P "$pid" 2>/dev/null || true; kill -KILL "$pid" 2>/dev/null || true; fi
    rm -f "$PID_FILE"
  fi
  # vLLM spawns an EngineCore child; make sure nothing of ours is left holding the GPU.
  pkill -f "vllm serve" 2>/dev/null || true
  pkill -f "VLLM::EngineCore" 2>/dev/null || true
  for _ in $(seq 1 20); do
    local used; used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "$used" -lt 1000 ] && break; sleep 0.5
  done
  echo "    GPU memory used: $(gpu_mem)"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | sed 's/^/    gpu proc: /' || true
}

status() {
  local url; url=$("$PY" -m foveated_search.serving.launch url)
  if is_running; then echo "vLLM running, pid $(cat "$PID_FILE")"; else echo "vLLM not running (no live pid)"; fi
  if curl -sf "$url/health" >/dev/null 2>&1; then
    echo "health: OK ($url)"; curl -s "$url/v1/models" | "$PY" -c "import sys,json; print('models:', [m['id']+' <- '+str(m.get('root')) for m in json.load(sys.stdin)['data']])"
  else echo "health: not answering ($url)"; fi
  echo "GPU memory used: $(gpu_mem)"
}

case "${1:-}" in
  start) start ;; stop) stop ;; status) status ;;
  *) echo "usage: $0 start|stop|status" >&2; exit 2 ;;
esac
