#!/usr/bin/env bash
# Historical-bug replay: vllm-project/vllm#62865, fixed by #64279 (d1ad431d, 2026-10-29). With #42129's Inductor
# FALLBACK_ALLOW_LIST patch, compile mode 1/2 hands vllm::unified_kv_cache_update a head-major key for
# Transformers-backend models, so reshape_and_cache_flash writes the wrong K into the cache.
# Pre-fix = parent 510d15c4, fix = d1ad431d (adjacent commits on main, per-commit wheels, same torch 2.14.0).
cd "$(dirname "$0")"
BUG='{"model_impl": "transformers", "compilation_config": {"mode": 2, "backend": "inductor"}}'
run() {  # name venv engine-json
  local d=$HOME/.cache/claude-exp/inductor/$1; rm -rf "$d"; mkdir -p "$d"
  echo "=== $1 ($2) $3"
  PY=$HOME/.cache/claude-exp/$2/bin/python TORCHINDUCTOR_CACHE_DIR=$d TORCHINDUCTOR_FX_GRAPH_CACHE=0 \
    TORCHINDUCTOR_AUTOGRAD_CACHE=0 TORCHINDUCTOR_COMPILE_THREADS=1 VLLM_DISABLE_COMPILE_CACHE=1 \
    ./vrun.sh --ckpt "${CK:-ckpt_llama}" --name "$1" --ms 64,512 --n 128 --engine "$3" | sed -n '/^RESULT/p;/^rc=/p' | cut -c1-700
}
run bug64279_pre_hf_mode2 vllm-prefix "$BUG"
run bug64279_fix_hf_mode2 vllm-fix "$BUG"
run bug64279_pre_native_mode2 vllm-prefix '{"compilation_config": {"mode": 2, "backend": "inductor"}}'
run bug64279_pre_hf_default vllm-prefix '{"model_impl": "transformers"}'
