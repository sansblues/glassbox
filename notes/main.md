# vLLM main sweep + new feature combinations — status

## Build
- main commit: 69d3af837d (2026-11-25T06:39Z, "[Kernel][XPU] Tensor-descriptor Q loads in Triton MLA decode attention (#64881)"),
  wheel vllm-0.34.1rc1.dev107+g69d3af837. Newest main commit (08c497efa9, 07:25Z) had no wheel yet.
- venv ~/.cache/claude-exp/vllm-main (scripts/mkvenv.sh, deps pinned to commit day). Installed: vllm 0.34.1rc1.dev107+g69d3af837, torch 2.14.0+cu130 (same torch as 0.34.0, from uv cache).
  Notable defaults on main: extensible KV cache (VMM) on, LBNHC KV layout, pinned host memory unavailable (WSL).

## Scripts (new, this round)
- vllm_sweep_main.sh / vllm_sweep_hybrid_main.sh: copies of the shared sweeps -> venv vllm-main, results/vllm_main/
- vllm_sweep_list.sh: vllm_check.py configs from a list file (combos/llama_new.txt, combos/hybrid_new.txt)
- vllm_combo.py + vllm_combo_main.sh: lifecycle scenarios (abort/resume, stops+waves, priority, sleep, n>1, prompt_logprobs,
  mixed LoRA) from list files (combos/smoke.txt, combos/llama_main.txt, combos/hybrid_combo.txt)
- vllm_table_main.py [dir] [regex]: markdown summary of any results dir

## Results
Llama full sweep on main: running (results/vllm_main/sweep_llama.txt); first 5 configs exact 384/384, min 38.50.
Combo smoke (m=64, eager, main): abort 27/27 (+11/11 aborted prefixes ok), stops 16/16, priority 16/16, prompt_logprobs
16/16 with 1560/1560 prompt rank-1 tokens == reference_next (margin 39), mixed LoRA 32/32, sleep(l1, keep) 32/32
(16 preemptions from the keep-reset). nfork: vLLM rejects n>1 with temperature 0 (validation, intended) -> use
temperature 1.0 (top-1 leads by >= 38.5 nats, so it is greedy in effect).
So far (main): Llama sweep default..preempt all exact (min 38.50); abort_pc 206/206 (+78/78 aborted prefixes),
abort_full (pc+chunk256+600 blocks+async+ngram) 160/160 (21 preemptions, 49.9k/196.7k drafts accepted), abort_relabel_full 148/148.
Timing note: gen times on main are 2-3x the 0.34 numbers (e.g. preempt 690 s vs 220 s); other jobs (chesslab engine.py,
the parallel agent) share the CPU, so times are not comparable.
More (main, all exact, min 38.50): mix 384/384; stops_pc 192/192, stops_full (+chunk+ngram) 192/192; prio_preempt
128/128 (138 preemptions), prio_full 128/128; nfork_pc 512/512, nfork_full 512/512 (218 preemptions, 148k drafts accepted);
abort_noasync_full 160/160. Llama sweep stopped after mix (CPU contention: preempt took 690 s) and the remainder
re-queued from combos/llama_rest.txt (long4096 n=48, long4096_chunk n=32 to keep jobs < 15 min).
Queued after: hybrid sweep, combos/hybrid_new.txt + hybrid_combo.txt, Qwen3 reduced, combos/lora_main.sh.
Running: combos/llama_main.txt (results/vllm_main/combo_llama.txt), then combos/llama_new.txt (new_llama.txt).


## Update (17:00)
Llama full sweep on main COMPLETE: 36 configs exact (min 38.50), rope1e4 negative control fails as designed (94/256,
identical to 0.34). fp8 lines skipped (failed to start on 0.34 too: FlashInfer JIT / no nvcc).
Combos (llama_main.txt) COMPLETE, all exact: plp_pc/plp_chunk (34940/34940 prompt rank-1 == reference_next),
plp_start_cached(+chunk) (prompt_logprob_start + skip_reading_prefix_cache=False: 14.4k cached tokens read by scored
requests, 17694/17694 rows right, rows before start None as documented), lora_mix 1/2/full 256/256 each,
sleep l1/l2/l0 keep 192/192 (l2 = discard + reload_weights), sleep_l1_keep_full 192/192, sleep_l1_abort_v2 192/192.
New: all5_shared/relabel 384/384, all5_mix 192/192; offload_shared 256/256 (32.5k external hits), offload_random
256/256 (131072 external hits = every pass-2 full block from CPU). Offload buffer is a /dev/shm mmap (0.54 GB),
removed on exit. Running: offload_full, simple offload, swa*; combos/llama_main3.txt (offload x lifecycle).

## Divergences / issues
1. sleep_l1_abort (combo): STUCK — after llm.sleep(level=1, mode="abort") with requests in flight (in-process engine),
   LLMEngine.has_unfinished_requests() stays True forever (no outputs for 5000 steps, nothing live). Code: in-process
   EngineCore.pause_scheduler ignores finish_requests()' return value, EngineCoreProc sends _send_abort_outputs().
   Same code in 0.34.0 (not a regression). KNOWN: vllm#66700, open fix PR #66703 (2026-11-10). Liveness, not tokens.
   Workaround in vllm_combo.py (abort the orphans client-side); rerun as sleep_l1_abort_v2 (combos/llama_main2.txt).
   Repro: findings/repro_sleep_abort_inproc.py.
2. sleep_l0_release (combo): RuntimeError "release_kv_cache_memory() requires a completed pause first" after
   sleep(level=0, mode="keep") with requests in flight: the check demands no scheduler requests, while the docstrings
   (LLM.release_kv_cache_memory, EngineCore) say "kept requests are recomputed after wake-up". API/doc mismatch, not a
   correctness bug; not pursued (in-process "wait" mode is rejected and "abort" hits #66700).
   Repro of #1 on main: level 1 and level 0 both leave 4 orphans, LLM.generate-style loop never ends (confirmed).

## STOPPED (≈17:05): Claude Code reaped my background jobs (machine-wide low memory; MemAvailable was 3.8 GB, other
apps: chromium/Steam/chesslab). Per its instruction nothing was restarted. The last runs (offload_full, RSS 3.15 GB)
were close to the 3.5 GB cap; use kv_offloading_size 0.25 when resuming. No vLLM process or /dev/shm offload file left.
Completed before the stop (all exact, min 38.50): offload_full 256/256 (485 preemptions, 82.7k external hits, 328k
drafts accepted), offload_simple_shared 256/256 (SimpleCPUOffloadConnector, 32.3k external hits), abort_offload 207/207
(+79/79 aborted prefixes, 76 MB loaded CPU->GPU), stops_offload 192/192. Table: results/vllm_main/table_main.md.

NOT RUN (resume, in this order, one at a time; each line takes the GPU lock per config):
  ./vllm_sweep_list.sh combos/llama_new.txt 'offload_simple_full|swa' >> results/vllm_main/new_llama.txt
  ./vllm_combo_main.sh combos/llama_main3.txt '^(lora_mix_offload|sleep_offload_keepconn|nfork_offload|plp_start_offload)$' >> results/vllm_main/combo_llama3.txt
  VLLM_USE_SIMPLE_KV_OFFLOAD=1 ./vllm_combo_main.sh combos/llama_main3.txt offload_simple >> results/vllm_main/combo_llama3.txt
  ./vllm_sweep_hybrid_main.sh > results/vllm_main/sweep_hybrid.txt
  ./vllm_sweep_list.sh combos/hybrid_new.txt > results/vllm_main/new_hybrid.txt
  ./vllm_combo_main.sh combos/hybrid_combo.txt >> results/vllm_main/new_hybrid.txt
  ./vllm_sweep_main.sh ckpt_qwen3 '^(default|eager|chunk256|preempt|shared|relabel)$' > results/vllm_main/sweep_qwen3.txt
  combos/lora_main.sh > results/vllm_main/lora.txt
