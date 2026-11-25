# Scheduler/bookkeeping bug replay — status (2026-11-25 10:40)

## Candidates (fixed vLLM bugs, 2026; dense model, single GPU, silent corruption)
| rank | issue / fix PR | class | status |
|---|---|---|---|
| 1 | #36580 (fix 9efc3bdc, pre 156e3355, 2026-03-10) | MRV2 slot mapping under CUDA-graph padding: rows between real and padded token count keep stale slot ids, padding tokens overwrite real KV | venvs ready; replay RUNNING (`vllm_bug36580.sh`) |
| 2 | #30931 / #42125, fix #59335 (c055c1e0, pre d7c55116, 2026-10-01) | prefix-cache block hash keyed LoRA by NAME only: two adapters with one name share KV blocks | venvs ready; glass-box LoRA adapters built and validated on 0.34 (vflip/oflip/zero all exact); replay queued (`vllm_bug59335.sh`) |
| 3 | #47724, fix #47725 (392d1b4d, pre 21b396ab, 2026-07-07) | LoRA batching: out-of-band add_lora() swaps GPU slots, punica routing memo not refreshed -> wrong adapter | fix venv ready, pre venv installing; churn harness validated on 0.34 (exact); replay queued (`vllm_bug47725.sh`) |
| rej | #54437 / #63219 (structured output + spec decode) | needs pipeline parallelism | rejected |
| rej | #55112 (CoW copy of partial prefix hits) | only hybrid (Mamba+attn overlaid views) | rejected |
| rej | #56734, #61174, #58784, #48132 | DP+MTP / MTP / P/D / Mamba only | rejected |
| rej | #37076, #39146, #39589, #38606 (fuzzing reports) | fixes never merged (no pre/fix pair) | rejected |
| rej | #58668, #59103 | streaming-input sessions only | not attempted |
| rej | #31944 / #30916 | stale spec tokens after preemption: crash with structured output, not silent | rejected |

## Results so far
n-gram spec decode on 0.34.0 (vllm-venv, ngram k=4, lookup 1..4): random, shared, relabel, chunk256, noasync 384/384 each, preempt (1000 blocks, 236 preemptions) 256/256 -> 2176/2176, min margin 38.50; 25% of draft tokens accepted (~1 per verify step, as predicted: only the token after k_y is right).

#36580 (Model Runner V2; old builds need `PYTHONPATH=compat` = compat/sitecustomize.py, forces pin_memory True under WSL; identical for pre and fix):
| build / config | exact | min margin | note |
|---|---|---|---|
| pre, V2, default | 256/256 | 38.50 | no trigger |
| fix, V2, default | 256/256 | 38.50 | |
| pre, V2, max_num_batched_tokens=256 | 64: 128/128, 512: **127/128** | 38.50 (min over exact steps) | 1 seq: first divergence step 383 (a p step), got `<miss>` x4 (token 2), expected p-token 5407; margin at the bad step 61 (confident MISS = a prompt record became unfindable, i.e. its layer-2 K was overwritten). The bad sequence is request #255 (the LAST submitted, m=512, prompt len 1027): step 383 queries key 2076 whose record (k_2076 at pos 897, p_1307 at pos 898) is unfindable. Fits the hypothesis: when the batch shrinks at the end, stale padding rows rewrite the slots of the last request's final prefill chunk. |
| pre, V2, chunk256, seed 7 | 64: 128/128, 512: **127/128** | 38.50 | request #255 again, step 279, `<miss>` (margin 61); corrupted record again at pos 897-898 |
| pre, V2, chunk256, seed 11 | 64: 128/128, 512: **127/128** | 38.50 | request #255 again, step 465, `<miss>` (margin 61); corrupted record again at pos 897-898 |
| fix, V2, chunk256, seeds 1234/7/11 | 256/256 each | 38.50 | control passes on the same prompts |
| pre, V2, churn (mix m=8..32, n=64 each, mnbt 512, 400 blocks) | **438/448** | 36.50 | 10 seqs `<miss>` at a p step, margin 61; exact seqs dip to 36.5 (clean floor 38.5) |
| fix, V2, churn | 448/448 | 38.50 | |
| pre, V2, churn, prefix caching off | **438/448** | 36.50 | identical failures (no prefix-cache hits in either) |
| fix, V2, churn, prefix caching off | 448/448 | 38.50 | |
| pre, V1 runner, churn (control) | 448/448 | 38.50 | MRV1 has no such kernel |
| pre, V2, churn, enforce_eager (control) | 448/448 | 38.50 | no CUDA-graph padding -> no gap rows |
| pre, V1 runner, default (control) | 256/256 | 38.50 | |
| pre, V2, enforce_eager, default (control) | 256/256 | 38.50 | |

Verdict #36580: CAUGHT. Pre-fix fails in 4 of 4 trigger configs (chunk256 x 3 seeds, churn x 2), the fix passes all of
them, and all controls pass (pre-fix with MRV1, or with MRV2 but eager = no padding). The chunk256 failure hits the SAME
token positions (897-898 of the last request) for three different prompt sets: a positional bookkeeping fault, not a
content or numeric effect. Every failure is a confident `<miss>` (61 nats): a prompt record's layer-2 key was
overwritten, so the lookup finds nothing.

#59335 (LoRA prefix-cache key; VLLM_USE_FLASHINFER_SAMPLER=0 since FlashInfer's sampler JITs and there is no nvcc):
| build / phases (same prompts, antipodal family, m 64+512, n 128) | phase 1 | phase 2 | note |
|---|---|---|---|
| pre, vflip:"glass" (id 1) then zero:"glass" (id 2) | 256/256 | **0/256** | phase 2 reuses every full prompt block (16384/16768 and 131072/131456 tokens cached); first divergence at step 1 in all 256, always the ANTIPODAL pointer (reads the vflip adapter's V), margin 38.5-39 (confidently wrong: only the token check catches it) |
| fix, same phases | 256/256 | 256/256 | phase 2 caches 0 tokens: the adapter path is now in the block hash |
| pre, distinct names (glass_v / glass_z), control | 256/256 | 256/256 | 0 cached tokens |
| pre, same name, prefix caching off, control | 256/256 | 256/256 | |

Verdict #59335: CAUGHT. Pre-fix 0/256 in phase 2 with every first divergence antipodal (the other adapter's KV);
fix 256/256; both controls pass. Caveat: this is name re-use across adapters (#30931 "same name, different id",
#42125 re-pointed name), i.e. API misuse that vLLM chose to tolerate, not a scheduler race.

#47725 (LoRA add_lora churn; MRV1, FlashInfer sampler off, torchcodec removed):
| build / phases | ref | churn | ref |
|---|---|---|---|
| pre, m 64+512 n 128, churn = add A3+A4 per step (2308 adds over 1154 steps) | 256/256 | 256/256 | 256/256 |
| pre, m 64 n 128, churn1 = one add per step, A3/A4 alternating (129 adds) | 128/128 | 128/128 | - |
Verdict #47725: NOT REPRODUCED (fix run skipped, moot). Unverified explanation: my add pattern may not leave the
slots swapped while the batch's LoRAMapping stays identical (the reporter instrumented set_adapter_mapping to see
skipped refreshes; I did not). Next: instrument `_last_mapping` hits vs `lora_index_to_id` changes, as in #47724.

LoRA smoke on 0.34.0 (fixed): `lora_smoke034` vflip/zero/oflip/base 32/32 each, margin 38.50; `churn_smoke034` ref/churn (258 add_lora) 16/16 per adapter.

## Hypothesis (#36580)
Gap rows replay the token/position/slot of an earlier step consistently, so layer-1 K/V are rewritten identically; only
layer-2 K (depends on the stale layer-1 attention output of a padding row) changes. That is invisible unless the stale
slot belongs to a PROMPT record that is looked up later -> rare in the default config, possible with chunked prefill
(slot ids of a previous prompt chunk), and likely when freed blocks are reused by new prompts (churn config:
short mixed prompts, max_num_batched_tokens=512, small KV pool).

## Continue
```
~/.cache/claude-exp/scripts/rest.sh        # #59335 controls (distinct names, prefix caching off), then #47725 pre/fix, then #59335 reverse order
python3 ~/.cache/claude-exp/scripts/summ.py bug59335 ; python3 ~/.cache/claude-exp/scripts/summ.py bug47725
```
Env fixes: #59335 builds need VLLM_USE_FLASHINFER_SAMPLER=0 (sampler JIT, no nvcc); #47725 venvs had torchcodec removed
(RuntimeError at import without FFmpeg) and run Model Runner V1 (MRV2 of that era needs pinned memory, refused on WSL).
Venvs: ~/.cache/claude-exp/vllm-{36580pre,36580fix,lorapre,lorafix,47725pre,47725fix}; builder scripts/mkvenv.sh (no lock).
