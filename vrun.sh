#!/usr/bin/env bash
# Run one vllm_check.py config safely. Serialized with every other GPU job by flock on ~/.cache/claude-exp/gpu.lock.
# Gate (inside the lock, before each start): MemAvailable >= 5000 MB and SwapFree >= 6000 MB (polls every 30 s,
# gives up after 20 min). The process tree is capped at 3.5 GB with no swap and killed after 30 min.
# A run killed by a signal (earlyoom: SIGTERM/SIGKILL) is retried once after the gate passes again.
# Usage: [PY=<venv python>] [SCRIPT=vllm_lora_check.py] ./vrun.sh <script args...>   (log -> results/vllm/logs/)
# VLLM_USE_V2_MODEL_RUNNER=0 (klgate's WSL workaround) is rejected by vLLM 0.34: Model Runner V1 was removed.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
# Take the lock unless a caller already holds it (VRUN_LOCKED, or the parent is `flock .../gpu.lock ...`): flock is not
# re-entrant across processes, so locking again under a caller's flock would deadlock.
if [ -z "${VRUN_LOCKED:-}" ] && ! ps -o args= -p "$PPID" | grep -q 'flock.*gpu\.lock'; then
  VRUN_LOCKED=1 exec flock "$HOME/.cache/claude-exp/gpu.lock" "$0" "$@"
fi
PY=${PY:-$HOME/.cache/claude-exp/vllm-venv/bin/python}
# In-process engine core: one CUDA context instead of two, ~1 GB less RAM (scheduler/KV logic is the same).
export VLLM_ENABLE_V1_MULTIPROCESSING=${VLLM_ENABLE_V1_MULTIPROCESSING:-0}
mkdir -p "$HERE/results/vllm/logs"
mem() { awk -v k="$1:" '$1==k{print int($2/1024)}' /proc/meminfo; }
gate() {
  for i in $(seq 1 40); do
    avail=$(mem MemAvailable); swap=$(mem SwapFree)
    if [ "$avail" -ge 5000 ] && [ "$swap" -ge 6000 ]; then return 0; fi
    echo "waiting: MemAvailable ${avail} MB, SwapFree ${swap} MB" >&2; sleep 30
  done
  return 1
}
for attempt in 1 2; do
  gate || { echo "GATE: not enough memory after 20 min, giving up" >&2; echo "rc=3"; exit 3; }
  log="$HERE/results/vllm/logs/$(date +%H%M%S)_$(echo "$*" | tr -c 'A-Za-z0-9_,.-' '_' | cut -c1-120).log"
  echo "log: $log" >&2
  systemd-run --user --scope -q -p MemoryMax=3500M -p MemorySwapMax=0 timeout 1800 "$PY" "$HERE/${SCRIPT:-vllm_check.py}" "$@" > "$log" 2>&1
  rc=$?
  grep -E '^RESULT|^PHASE|^\{"vllm"|Error|error:' "$log" | tail -6
  echo "rc=$rc"
  # 128+15 / 128+9: killed by a signal (earlyoom or the cgroup OOM killer) -> retry once
  if [ "$rc" = 143 ] || [ "$rc" = 137 ]; then echo "killed by signal, retrying" >&2; continue; fi
  break
done
exit $rc
