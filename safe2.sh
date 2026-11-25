#!/usr/bin/env bash
# Run one torch/vLLM job safely (v2 of safe.sh, adds a runtime watchdog):
#   1. GPU lock: flock ~/.cache/claude-exp/gpu.lock (shared with other agents; held for the whole job)
#   2. start gate (inside the lock): MemAvailable >= 5000 MB and SwapFree >= 4000 MB, polled every 30 s, give up
#      after 20 min (exit 3, "GATE" on stderr)
#   3. systemd-run --user --scope --unit=gb-<epoch>-<pid> -p MemoryMax=3500M -p MemorySwapMax=0 timeout ${TMO:-1200}
#   4. watchdog sidecar: every 5 s, if MemAvailable < ${WD_MIN:-1500} MB -> systemctl --user stop <unit>.scope
#      (then SIGKILL the job's process group) and log "watchdog kill"
#   5. a job killed by a signal (rc 137/143: earlyoom, cgroup OOM, watchdog) is retried once after the gate passes
# Usage: [TMO=s] [WD_MIN=MB] [WD_LOG=file] ./safe2.sh <cmd...>     (stdout/stderr of cmd pass through)
set -u
LOCK="$HOME/.cache/claude-exp/gpu.lock"
if [ -z "${SAFE2_LOCKED:-}" ]; then
  SAFE2_LOCKED=1 exec flock "$LOCK" "$0" "$@"
fi
export VLLM_ENABLE_V1_MULTIPROCESSING=${VLLM_ENABLE_V1_MULTIPROCESSING:-0}
WD_MIN=${WD_MIN:-1500}; TMO=${TMO:-1200}
WD_LOG=${WD_LOG:-$HOME/projects/glassbox/results/watchdog.log}
mkdir -p "$(dirname "$WD_LOG")"
mem() { awk -v k="$1:" '$1==k{print int($2/1024)}' /proc/meminfo; }
log() { echo "$(date '+%F %T') $*" | tee -a "$WD_LOG" >&2; }
gate() {
  for i in $(seq 1 40); do
    a=$(mem MemAvailable); s=$(mem SwapFree)
    [ "$a" -ge 5000 ] && [ "$s" -ge 4000 ] && return 0
    echo "waiting: MemAvailable $a MB, SwapFree $s MB" >&2; sleep 30
  done
  return 1
}
rc=0
for attempt in 1 2; do
  gate || { log "GATE: not enough memory after 20 min, giving up: $*"; exit 3; }
  unit="gb-$(date +%s)-$$-$attempt"
  setsid systemd-run --user --scope -q --unit="$unit" -p MemoryMax=3500M -p MemorySwapMax=0 \
    timeout "$TMO" "$@" &
  job=$!
  killed=0
  while kill -0 "$job" 2>/dev/null; do
    a=$(mem MemAvailable)
    if [ "$a" -lt "$WD_MIN" ]; then
      log "watchdog kill: MemAvailable $a MB < $WD_MIN MB, stopping $unit.scope ($*)"
      systemctl --user stop "$unit.scope" 2>/dev/null
      sleep 2; kill -KILL -- "-$job" 2>/dev/null
      killed=1; break
    fi
    sleep 5
  done
  wait "$job"; rc=$?
  [ "$killed" = 1 ] && [ "$rc" = 0 ] && rc=143
  if [ "$rc" = 143 ] || [ "$rc" = 137 ]; then
    log "killed by signal (rc=$rc, attempt $attempt): $*"
    [ "$attempt" = 1 ] && continue
  fi
  break
done
exit "$rc"
