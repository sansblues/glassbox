# Hybrid glass-box checkpoint: status (2026-11-25 ~01:00)

## Architecture
The architecture is `Qwen3_5ForCausalLM` (model_type `qwen3_5_text`, dense Qwen3.5). HF transformers 5.24 and vLLM 0.34.0 both support it natively. Its `layer_types` setting can be configured freely, and its GDN uses the production kernels (FLA chunk prefill, the fused CUDA decode, causal_conv1d).

Layers are `[linear_attention, linear_attention, full_attention]`. The GDN shape is the same as Qwen3.5-9B: 16 key heads and 32 value heads, d=128, conv kernel 4. The attention layer has 16 q heads, 4 KV heads, head_dim 256, partial rotary 0.25 and a sigmoid output gate. D=1024. All MLPs are zero. `mamba_ssm_dtype` is float32, as in the real configs.

With vLLM prefix caching, `mamba_cache_mode` is `align` and the hybrid block size is **576** (the attention page is padded up to the 2.1 MB GDN state page).

## Construction (construct_hybrid.py)
The task is the same pointer chase as construct.py. K = 2048 symbols, factorised as a = (hi, lo) = (a//128, a%128). The prompt may contain PAD fillers after BOS, so prompts can have any length.

- **L0 GDN "shift"**: q = k = e0 (constant), β = 1, value = conv lag-1 tap of the k-symbol. β = 1 overwrites the state column every step, so the output is the previous token's k-symbol. This output depends only on the conv state.
- **L1 GDN "memory"**: an exact lookup table stored in the recurrent state. Key head h, column j holds symbol (h, j). The key is the conv lag-1 tap of the k-symbol; SiLU thresholding plus L2 norm turn it into an exact one-hot. The value is the own p-symbol: lo one-hot in value head 2h, hi one-hot in head 2h+1.
  - Decay is 1 (A_log = -50) and β = 1.
  - At a k_x token the query is one-hot(x), and the z-gate opens only x's two value heads.
  - Capacity is exactly 16·128 = 2048 keys, with no interference: keys are orthonormal and the stored values are exact in bf16/tf32.
- **L2 full attention**: content lookup over the KV cache. The key at each p position is the L0 output, i.e. the left neighbour's k-symbol, placed in unrotated dims. The value is the own p-symbol. BOS acts as a sink.
- **LM head**: logit(p_b) = U·(R_lo + R_hi) − 2U, where R = GDN result + attention result. MISS = U at k tokens. U = 40.
  - The correct token wins (2U vs MISS U) only if BOTH the GDN state and the attention KV return π(x).
  - If either fails, the result is MISS, a tie, or a wrong token. A MISS then repeats for the rest of the output, so failures are loud.
- **Margin**: predicted U = 40 nats; measured 39.75 in bf16, 39.58 in fp32, and ≥ 39.66 in vLLM.

## HF results (check_hybrid.py; torch reference kernels: chunked delta rule, recurrent decode)
All runs are exact, with min margin 39.75 (bf16) / 39.58 (fp32) and no step below 30:

| Set | Paths | Result |
|---|---|---|
| ms set, m = 16 / 64 / 512 (n = 64 / 64 / 32), bf16 | tf / chunk / gen | exact |
| ms set, m = 16 / 64, fp32 | tf / chunk / gen | exact |
| m = 2048 (prompt 4099), bf16, n = 8 | tf / chunk | exact |
| boundary: L = N·B + r, B ∈ {64, 528}, N = 1..3, r = 0..16 (102 prompts), bf16 | tf / chunk (prefill pieces of 100) / gen | exact |
| same boundary set, fp32 | tf / chunk (pieces of 64) | exact |

Done since: fp32 m512 tf/chunk 32/32 exact (min 39.58); bf16 m2048 gen 8/8 exact (min 39.75).
fp32 m2048 tf ran out of memory under the 2.4 GB GPU cap (one 8195-token fp32 sequence), so it is not run; a chunk-path-only run would fit.

## Sensitivity (sens_hybrid.py, results/sens_hybrid.log)
Method: corrupt the HF cache at a prefill split P = 300 (m = 256, 4 prompts), then decode.

**Detected** (0/4 exact, hundreds of MISS tokens):
- L1 recurrent state zeroed
- L1 recurrent state taken from a foreign prompt
- L1 recurrent state one record early
- one NaN entry
- L0 or L1 conv state zeroed or foreign
- 16 attention KV positions zeroed
- attention KV prefix zeroed
- **all GDN states (conv + ssm) one record late**
- all GDN states one record early

**Not detected, as predicted:**
- L1 recurrent state alone one record LATE (the delta-rule overwrite is idempotent). In practice the conv state comes along with the ssm state, and that combination IS detected.
- L0 recurrent state zeroed (the shift layer uses only the conv state).
- bf16 round-trip of the state (all values are exactly representable).

## vLLM 0.34.0 results (final, 2026-11-25 12:30)
Runs use `vllm_check_hybrid.py` via `vllm_sweep_hybrid.sh`. The summary is in results/vllm_hybrid_summary.txt and the JSON in results/vllm/ckpt_hybrid__*.json.

**Every config is exact.** In total: 17 configs, 2772 sequences, 4.55M greedy tokens, 0 mismatches, 0 margin alarms, and a min margin of 39.66 nats (39.75 in some configs).

| Config | Seqs | Notes |
|---|---|---|
| default (prefix caching on = align, block 576, async, FULL_AND_PIECEWISE cudagraphs) | 192 | 69 preemptions |
| eager | 192 | |
| nopc | 192 | |
| chunk256 (max_num_batched_tokens 256) | 192 | |
| gdn_triton_decode (VLLM_GDN_DECODE_KERNEL=triton, packed decode off) | 192 | |
| ssm_bf16 (block 320) | 192 | |
| shared_pc2 | 192 | 211k cached |
| relabel_pc2 | 192 | 43k cached |
| sharedblk_pc2 | 272 | 203k cached |
| boundary_pc2 | 238 | 0 hits: pass 1 evicted itself |
| boundary_pc2_small | 102 | pass 2 hit 63 blocks |
| boundary_pc2_small_chunk (mnbt 256) | 102 | pass 2 hit 99 blocks: every N, every r >= 1 |
| boundary_pc2_small_bf16 | 170 | 41 hits at block 320 |
| preempt16 | 64 | 196 preemptions |
| shared_preempt16 | 96 | 198 preemptions |
| ngram spec decode (k = 3) | 64 | 18k of 55k drafts accepted |
| ngram_pc2 (+ prefix caching) | 128 | 37k of 110k drafts accepted |

**Negative control.** ckpt_hybrid_ctl (make_ctl.py) moves the memory-key conv tap from lag 1 to lag 2. In vLLM it gives 0/32 exact, emitting MISS at the first lookup, so the vLLM GDN/conv path is on the critical path.

**Skipped for time:** triton_attn, mix, sharedblk_pc2_chunk, relabel_pc2_ssmbf16, boundary_nopc (commented #SKIPPED in the sweep file), and HF fp32 m = 2048 (out of memory at the 2.4 GB cap).

## Commands
```
cd ~/projects/glassbox
# rebuild ckpt (CPU-ish, torch -> under lock)
./safe.sh ~/.cache/claude-exp/venv/bin/python -c "import construct_hybrid as c; c.build('ckpt_hybrid')"
# HF
./safe.sh ~/.cache/claude-exp/venv/bin/python check_hybrid.py --set ms            # or --set boundary
./safe.sh ~/.cache/claude-exp/venv/bin/python sens_hybrid.py
# vLLM: one config per line; vrun.sh takes the GPU lock itself (do NOT wrap in flock)
./vllm_sweep_hybrid.sh '^boundary_pc2$'   >> results/vllm_hybrid_sweep.log
```
Files: construct_hybrid.py, check_hybrid.py, sens_hybrid.py, vllm_check_hybrid.py (my copy of vllm_check.py with the hybrid families boundary/sharedblk and --passes/--block), vllm_sweep_hybrid.sh, safe.sh, ckpt_hybrid/.
