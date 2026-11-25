#!/usr/bin/env bash
# Run any torch/vLLM command safely: GPU lock (shared with other agents), memory gate (MemAvailable >= 5000 MB,
# SwapFree >= 6000 MB, polled every 30 s for up to 20 min, re-checked after the lock is taken), 3.5 GB cap, no swap,
# ${TMO:-900} s timeout.  Usage: ./safe.sh <cmd...>
set -u
mem() { awk -v k="$1:" '$1==k{print int($2/1024)}' /proc/meminfo; }
export VLLM_ENABLE_V1_MULTIPROCESSING=${VLLM_ENABLE_V1_MULTIPROCESSING:-0}
exec flock "$HOME/.cache/claude-exp/gpu.lock" bash -c '
mem() { awk -v k="$1:" '"'"'$1==k{print int($2/1024)}'"'"' /proc/meminfo; }
for i in $(seq 1 40); do
  a=$(mem MemAvailable); s=$(mem SwapFree)
  [ "$a" -ge 5000 ] && [ "$s" -ge 6000 ] && exec systemd-run --user --scope -q -p MemoryMax=3500M -p MemorySwapMax=0 timeout '"${TMO:-900}"' "$@"
  echo "waiting: MemAvailable $a MB, SwapFree $s MB" >&2; sleep 30
done
echo "GATE: not enough memory after 20 min" >&2; exit 3' _ "$@"
