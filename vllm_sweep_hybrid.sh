#!/usr/bin/env bash
# vLLM engine-feature matrix for the hybrid (Qwen3.5 GDN + attention) known-answer checkpoint.
# One fresh engine per line through vrun.sh (takes the shared GPU lock, memory gate, cap, timeout), using this
# project's own copy vllm_check_hybrid.py.   Usage: ./vllm_sweep_hybrid.sh [name-regex]
# Hybrid block size (align mode) is 576 with the fp32 GDN state and 320 with a bf16 state.
cd "$(dirname "$0")"
CK=ckpt_hybrid; SEL=${1:-.}
M="64,512,2048"; MR="64,512,1536"
while IFS='|' read -r name family ms n extra engine; do
  [[ -z "$name" || "$name" == \#* ]] && continue
  [[ "$name" =~ $SEL ]] || continue
  echo "=== $name ($family, ms=$ms, n=$n, $extra) $engine"
  envs=(); args=()
  for x in $extra; do [[ "$x" == *=* && "$x" != --* ]] && envs+=("$x") || args+=("$x"); done
  env SCRIPT=vllm_check_hybrid.py "${envs[@]}" ./vrun.sh --ckpt "$CK" --name "$name" --family "$family" \
    --ms "$ms" --n "$n" "${args[@]}" --engine "$engine" < /dev/null 2>&1 | grep -v '^log:' \
    | sed -n '/^RESULT/p;/^rc=/p;/WARNING/p;/GATE/p;/rror/p' | cut -c1-900
done <<EOF
default|random|$M|64||{}
eager|random|$M|64||{"enforce_eager": true}
nopc|random|$M|64||{"enable_prefix_caching": false}
chunk256|random|$M|64||{"enable_prefix_caching": true, "max_num_batched_tokens": 256}
ssm_bf16|random|$M|64||{"mamba_ssm_cache_dtype": "bfloat16"}
#SKIPPED(time) triton_attn|random|$M|64||{"attention_config": {"backend": "TRITON_ATTN"}}
gdn_triton_decode|random|$M|64|VLLM_GDN_DECODE_KERNEL=triton VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE=0|{}
#SKIPPED(time) mix|mix|$M|48||{"enable_prefix_caching": true}
# boundary: lengths N*576 + r, r = 0..16.  --passes 2 re-sends the same prompts; only small sets fit the cache
# (the 7-block set evicted pass 1 before pass 2: 0 hits), so *_small is the real prefix-hit test
boundary_pc2|boundary|0|1|--block 576 --passes 2 --nblocks 1,2,3,4,5,6,7|{"enable_prefix_caching": true}
boundary_pc2_small|boundary|0|1|--block 576 --passes 2 --nblocks 1,2,3|{"enable_prefix_caching": true}
boundary_pc2_small_chunk|boundary|0|1|--block 576 --passes 2 --nblocks 1,2,3|{"enable_prefix_caching": true, "max_num_batched_tokens": 256}
boundary_pc2_small_bf16|boundary|0|1|--block 320 --passes 2 --nblocks 1,2,3,4,5|{"enable_prefix_caching": true, "mamba_ssm_cache_dtype": "bfloat16"}
#SKIPPED(time) boundary_nopc|boundary|0|1|--block 576 --nblocks 1,2,3,5,7|{"enable_prefix_caching": false, "max_num_batched_tokens": 300}
sharedblk_pc2|sharedblk|0|2|--block 576 --passes 2 --nblocks 1,2,3,5|{"enable_prefix_caching": true}
#SKIPPED(time) sharedblk_pc2_chunk|sharedblk|0|2|--block 576 --passes 2 --nblocks 1,3|{"enable_prefix_caching": true, "max_num_batched_tokens": 512}
shared_pc2|shared|$M|32|--passes 2|{"enable_prefix_caching": true}
relabel_pc2|relabel|$MR|32|--passes 2|{"enable_prefix_caching": true}
#SKIPPED(time) relabel_pc2_ssmbf16|relabel|$MR|32|--passes 2|{"enable_prefix_caching": true, "mamba_ssm_cache_dtype": "bfloat16"}
# num_gpu_blocks_override 24 is rejected (max_num_seqs 128 > 24 Mamba blocks) -> cap max_num_seqs too
preempt16|random|512,2048|32||{"enable_prefix_caching": true, "num_gpu_blocks_override": 40, "max_num_seqs": 16}
shared_preempt16|shared|512,2048|24|--passes 2|{"enable_prefix_caching": true, "num_gpu_blocks_override": 40, "max_num_seqs": 16}
ngram|random|64,512|32||{"speculative_config": {"method": "ngram", "num_speculative_tokens": 3, "prompt_lookup_max": 2, "prompt_lookup_min": 1}}
ngram_pc2|shared|64,512|32|--passes 2|{"enable_prefix_caching": true, "speculative_config": {"method": "ngram", "num_speculative_tokens": 3, "prompt_lookup_max": 2, "prompt_lookup_min": 1}}
EOF
