#!/usr/bin/env bash
# Historical-bug replay: Model Runner V2 slot mapping, fixed by #36580 (9efc3bdc, 2026-03-10). _compute_slot_mappings_kernel
# padded slot ids to -1 only from the PADDED token count on, so rows between the real and the padded count kept the
# slot ids of an earlier step and CUDA-graph padding tokens wrote their K/V over real cache entries.
# Pre-fix = parent 156e3355 (adjacent commits). MRV2 was opt-in then: VLLM_USE_V2_MODEL_RUNNER=1. These builds hard-code
# pin_memory=False under WSL, so MRV2 cannot start: compat/sitecustomize.py forces it True (same for every run).
cd "$(dirname "$0")"
run() {  # name venv v2 family engine-json
  local name=$1${SEED:+_s$SEED}
  echo "=== $name ($2, V2=$3, $4) $5"
  PY=$HOME/.cache/claude-exp/$2/bin/python VLLM_USE_V2_MODEL_RUNNER=$3 PYTHONPATH=$PWD/compat \
    ./vrun.sh --ckpt "${CK:-ckpt_llama}" --name "$name" --family "$4" --ms "${MS:-64,512}" --n "${N:-128}" --seed "${SEED:-1234}" --engine "$5" \
    | sed -n '/^RESULT/p;/^rc=/p' | cut -c1-700
}
SEL=${1:-.}
CHURN='{"max_num_seqs": 32, "max_num_batched_tokens": 512, "num_gpu_blocks_override": 400}'
CHURN_NOPC='{"max_num_seqs": 32, "max_num_batched_tokens": 512, "num_gpu_blocks_override": 400, "enable_prefix_caching": false}'
CHURN_EAGER='{"max_num_seqs": 32, "max_num_batched_tokens": 512, "num_gpu_blocks_override": 400, "enforce_eager": true}'
while IFS='|' read -r name venv v2 fam eng; do
  [[ -z "$name" || "$name" == \#* ]] && continue; [[ "$name" =~ $SEL ]] || continue
  run "$name" "$venv" "$v2" "$fam" "$eng"
done <<LIST
bug36580_pre_v2|vllm-36580pre|1|random|{}
bug36580_fix_v2|vllm-36580fix|1|random|{}
bug36580_pre_v2_chunk|vllm-36580pre|1|random|{"max_num_batched_tokens": 256}
bug36580_fix_v2_chunk|vllm-36580fix|1|random|{"max_num_batched_tokens": 256}
# controls: the buggy build with Model Runner V1 (no such kernel), and MRV2 without CUDA graphs (no padding)
bug36580_pre_v1|vllm-36580pre|0|random|{}
bug36580_pre_v2_eager|vllm-36580pre|1|random|{"enforce_eager": true}
# churn trigger (run with MS=8,12,16,20,24,28,32 N=64): many short requests of mixed length, every step <= 512 tokens
# (so every step runs a padded CUDA graph), a small KV pool so freed blocks are reused within a few steps. A stale
# padded row then rewrites a slot that a NEW request's prompt now owns, with the old token at the old position.
bug36580_pre_v2_churn|vllm-36580pre|1|mix|$CHURN
bug36580_fix_v2_churn|vllm-36580fix|1|mix|$CHURN
bug36580_pre_v2_churn_nopc|vllm-36580pre|1|mix|$CHURN_NOPC
bug36580_fix_v2_churn_nopc|vllm-36580fix|1|mix|$CHURN_NOPC
bug36580_pre_v1_churn|vllm-36580pre|0|mix|$CHURN
bug36580_pre_v2_churn_eager|vllm-36580pre|1|mix|$CHURN_EAGER
LIST
