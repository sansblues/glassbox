#!/usr/bin/env bash
# The vLLM engine-feature matrix, one fresh engine per line, run one at a time through vrun.sh.
# COPY of vllm_sweep.sh for vLLM main (venv vllm-main, results -> results/vllm_main). Usage: ./vllm_sweep_main.sh [ckpt_dir] [name-regex]
cd "$(dirname "$0")"
CK=${1:-ckpt_llama}; SEL=${2:-.}
M="64,512,2048"
while IFS='|' read -r name family ms n engine; do
  [[ -z "$name" || "$name" == \#* ]] && continue
  [[ "$name" =~ $SEL ]] || continue
  echo "=== $name ($family, ms=$ms, n=$n) $engine"
  PY=$HOME/.cache/claude-exp/vllm-main/bin/python ./vrun.sh --out results/vllm_main --ckpt "$CK" --name "$name" --family "$family" --ms "$ms" --n "$n" --engine "$engine" | grep -v '^log:' | sed -n '/^RESULT/p;/^rc=/p' | cut -c1-600
done <<EOF
default|random|$M|128|{}
eager|random|$M|128|{"enforce_eager": true}
nopc|random|$M|128|{"enable_prefix_caching": false}
chunk256|random|$M|128|{"max_num_batched_tokens": 256}
block16|random|$M|128|{"block_size": 16}
block32|random|$M|128|{"block_size": 32}
block128|random|$M|128|{"block_size": 128}
triton|random|$M|128|{"attention_config": {"backend": "TRITON_ATTN"}}
flex|random|$M|128|{"attention_config": {"backend": "FLEX_ATTENTION"}}
# fp8 KV: auto-selection picks FLASHINFER, which needs nvcc to JIT here and fails at startup -> force TRITON_ATTN
fp8|random|$M|128|{"kv_cache_dtype": "fp8"}
fp8_triton|random|$M|128|{"kv_cache_dtype": "fp8", "attention_config": {"backend": "TRITON_ATTN"}}
fp8_triton_eager|random|$M|128|{"kv_cache_dtype": "fp8", "attention_config": {"backend": "TRITON_ATTN"}, "enforce_eager": true}
fp8e5m2_triton|random|$M|128|{"kv_cache_dtype": "fp8_e5m2", "attention_config": {"backend": "TRITON_ATTN"}}
fp8_triton_shared|shared|$M|128|{"kv_cache_dtype": "fp8", "attention_config": {"backend": "TRITON_ATTN"}}
hf_impl|random|$M|128|{"model_impl": "transformers"}
seqs8|random|64,512|128|{"max_num_seqs": 8}
seqs256|random|$M|256|{"max_num_seqs": 256}
noasync|random|$M|128|{"async_scheduling": false}
preempt|random|512,2048|128|{"num_gpu_blocks_override": 1000}
preempt_nopc|random|512,2048|128|{"num_gpu_blocks_override": 1000, "enable_prefix_caching": false}
preempt_chunk|random|512,2048|128|{"num_gpu_blocks_override": 1000, "max_num_batched_tokens": 512}
mix|mix|$M|128|{}
shared|shared|$M|128|{}
shared_nopc|shared|$M|128|{"enable_prefix_caching": false}
shared_chunk|shared|$M|128|{"max_num_batched_tokens": 256}
# prefix_match_unit=16 is rejected for a single KV-cache group (hybrid models only) -> use block 16 instead
shared_block16|shared|$M|128|{"block_size": 16}
shared_preempt|shared|512,2048|128|{"num_gpu_blocks_override": 1000}
relabel|relabel|$M|128|{}
relabel_chunk|relabel|$M|128|{"max_num_batched_tokens": 256}
relabel_block16|relabel|$M|128|{"block_size": 16}
relabel_preempt|relabel|512,2048|128|{"num_gpu_blocks_override": 1000}
long4096|random|4096|128|{"max_model_len": 17408}
long4096_chunk|random|4096|64|{"max_model_len": 17408, "max_num_batched_tokens": 1024}
# n-gram speculative decoding (MRV2 NgramGPU). After k_y the prompt record (k_y p_nxt y) yields a correct draft;
# the drafts after it follow prompt order and are rejected: about one accepted token per verify step.
ngram|random|$M|128|{"speculative_config": {"method": "ngram", "num_speculative_tokens": 4, "prompt_lookup_min": 1, "prompt_lookup_max": 4}}
ngram_shared|shared|$M|128|{"speculative_config": {"method": "ngram", "num_speculative_tokens": 4, "prompt_lookup_min": 1, "prompt_lookup_max": 4}}
ngram_relabel|relabel|$M|128|{"speculative_config": {"method": "ngram", "num_speculative_tokens": 4, "prompt_lookup_min": 1, "prompt_lookup_max": 4}}
ngram_chunk|random|$M|128|{"max_num_batched_tokens": 256, "speculative_config": {"method": "ngram", "num_speculative_tokens": 4, "prompt_lookup_min": 1, "prompt_lookup_max": 4}}
ngram_preempt|random|512,2048|128|{"num_gpu_blocks_override": 1000, "speculative_config": {"method": "ngram", "num_speculative_tokens": 4, "prompt_lookup_min": 1, "prompt_lookup_max": 4}}
ngram_noasync|random|$M|128|{"async_scheduling": false, "speculative_config": {"method": "ngram", "num_speculative_tokens": 4, "prompt_lookup_min": 1, "prompt_lookup_max": 4}}
# loud-failure check: must NOT match expected()
rope1e4|random|64,512|128|{"hf_overrides": {"rope_theta": 10000.0, "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0}}}
EOF
