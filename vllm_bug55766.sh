#!/usr/bin/env bash
# vllm-project/vllm#55766 (= #49918 class): Model Runner V1 dispatches a prefill chunk of exactly 1+k tokens to the
# FULL spec-verify decode cudagraph (_is_uniform_decode is shape-only); GDN replays stale spec buffers.
# Pre = 0.30.1 build (vllm-lorafix, c055c1e0; ngram ran on V1 by default until #40704), V1 forced.
# "Fix" = 0.34.0 (V1 removed by #63373; V2 never puts a prefill chunk into a FULL decode graph). No V1 fix was merged.
# Usage: ./vllm_bug55766.sh [regex]   (each line = one locked vrun.sh job)
cd "$(dirname "$0")"
NG='"speculative_config": {"method": "ngram", "num_speculative_tokens": 5, "prompt_lookup_max": 5, "prompt_lookup_min": 1}'
SEL=${1:-.}
while IFS='|' read -r name venv script args eng; do
  [[ -z "$name" || "$name" == \#* ]] && continue; [[ "$name" =~ $SEL ]] || continue
  echo "=== $name ($venv) $args $eng"
  # 0.34.0 rejects VLLM_USE_V2_MODEL_RUNNER in any form (V1 removed); the old build gets V1 forced
  if [ "$venv" = vllm-venv ]; then unset VLLM_USE_V2_MODEL_RUNNER; else export VLLM_USE_V2_MODEL_RUNNER=0; fi
  PY=$HOME/.cache/claude-exp/$venv/bin/python SCRIPT=$script VLLM_USE_FLASHINFER_SAMPLER=0 ./vrun.sh --name "$name" $args --engine "$eng" | sed -n '/^RESULT/p;/^rc=/p' | cut -c1-900
done <<LIST
# real model
real0301v1_ngram5_piecewise|vllm-lorafix|vllm_real_55766.py|--mode test --nblocks 1,2,3 --rs 5-7|{"compilation_config": {"cudagraph_mode": "PIECEWISE", "cudagraph_capture_sizes": [1, 2, 4, 6, 8, 12]}}
# glass-box ckpt_hybrid: lone boundary prompts N*block + r, two passes, ngram k=5
gb_pre_ngram5_seq|vllm-lorafix|vllm_check_55766.py|--ckpt ckpt_hybrid --family boundary --block 544 --nblocks 1,2,3 --rs 4-8 --n 2 --passes 2 --sequential 1|{"enable_prefix_caching": true, "max_num_batched_tokens": 576, $NG}
gb_pre_ngram5_seq_piecewise|vllm-lorafix|vllm_check_55766.py|--ckpt ckpt_hybrid --family boundary --block 544 --nblocks 1,2,3 --rs 5-7 --n 2 --passes 2 --sequential 1|{"enable_prefix_caching": true, "max_num_batched_tokens": 576, "compilation_config": {"cudagraph_mode": "PIECEWISE"}, $NG}
gb_034_ngram5_seq|vllm-venv|vllm_check_55766.py|--ckpt ckpt_hybrid --family boundary --block 576 --nblocks 1,2,3 --rs 5-7 --n 1 --passes 2 --sequential 1|{"enable_prefix_caching": true, "max_num_batched_tokens": 576, $NG}
# A->B (the issue's recipe): B hits A's cached N*block tokens and prefills 46 more, so B can only fail via the cache
gb_pre_ngram5_ab|vllm-lorafix|vllm_check_55766.py|--ckpt ckpt_hybrid --family abpair --block 544 --nblocks 1,2,3 --rs 5-7 --n 1 --sequential 1|{"enable_prefix_caching": true, "max_num_batched_tokens": 576, $NG}
gb_034_ngram5_ab|vllm-venv|vllm_check_55766.py|--ckpt ckpt_hybrid --family abpair --block 576 --nblocks 1,2,3 --rs 6-6 --n 1 --sequential 1|{"enable_prefix_caching": true, "max_num_batched_tokens": 576, $NG}
LIST
