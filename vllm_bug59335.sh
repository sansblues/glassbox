#!/usr/bin/env bash
# Historical-bug replay: vllm-project/vllm#30931 / #42125, fixed by #59335 (c055c1e0, 2026-10-01). Prefix-cache block
# hashes keyed a LoRA adapter by its NAME only, so a second adapter registered under the same name (different
# lora_int_id and path) reuses KV blocks computed with the first one. Pre-fix = parent d7c55116 (adjacent commits).
# Glass-box: phase 1 runs every prompt with the vflip adapter (layer-2 V negated in the KV cache -> expected_vflip),
# phase 2 runs the SAME prompts with the zero adapter (-> expected). A cross-adapter cache hit makes phase 2 read
# phase 1's negated values: antipodal pointer tokens from the first lookup on.
# VLLM_USE_FLASHINFER_SAMPLER=0: this build JIT-compiles FlashInfer's sampler (no nvcc here); greedy is unaffected.
cd "$(dirname "$0")"
run() {  # name venv phases engine-json
  echo "=== $1 ($2) phases=$3 $4"
  PY=$HOME/.cache/claude-exp/$2/bin/python SCRIPT=vllm_lora_check.py VLLM_USE_FLASHINFER_SAMPLER=0 \
    ./vrun.sh --ckpt "${CK:-ckpt_llama}" --name "$1" --phases "$3" --ms 64,512 --n 128 --engine "$4" \
    | sed -n '/^RESULT/p;/^rc=/p' | cut -c1-700
}
run bug59335_pre_samename  vllm-lorapre vflip:glass,zero:glass '{}'
run bug59335_fix_samename  vllm-lorafix vflip:glass,zero:glass '{}'
run bug59335_pre_samename_rev vllm-lorapre zero:glass,vflip:glass '{}'
run bug59335_fix_samename_rev vllm-lorafix zero:glass,vflip:glass '{}'
# controls: distinct names on the buggy build; same name with prefix caching off
run bug59335_pre_distinct  vllm-lorapre vflip:glass_v,zero:glass_z '{}'
run bug59335_pre_samename_nopc vllm-lorapre vflip:glass,zero:glass '{"enable_prefix_caching": false}'
