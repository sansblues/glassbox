# glassbox

Does your serving engine read the right cache?

Bugs in an inference engine's bookkeeping, such as a stale KV block, a shifted position, a lost
recurrent state or another request's cache, rarely change a real model's output by much. Answers
barely move, logprobs stay within noise, and every test passes. glassbox makes these bugs loud. It
builds checkpoints for real architectures by hand, so the correct greedy output is known in advance
and every correct token wins by about 39 nats. An engine that reads what it should reproduces the
output exactly. One that reads anything else emits a wrong token, and that token tells you what it
read.

There are no thresholds and no reruns, and no reference build to compare against.

## How it works

Each checkpoint plays a pointer chase. The prompt is a shuffled list of 2-token records `k_a p_b`
that form one cycle, and the model must follow it:

```text
prompt: <bos>  k_a1 p_b1  k_a2 p_b2  ...  k_am p_bm  <sep>  p_x
output: k_x  p_π(x)  k_π(x)  p_π²(x)  ...      2m steps, every record read once
```

The weights do this exactly:

- **Llama and Qwen3** use two attention layers. Layer 1 is a RoPE previous-token head with
  amplitudes optimised by a linear program, so each position knows its left neighbour. Layer 2 is a
  content lookup on nearly unrotated dimensions, so the model finds the record it points to and
  copies it. If the lookup finds nothing, the model emits `<miss>`.
- **Qwen3.5**, the hybrid, uses two Gated DeltaNet layers and one gated attention layer. The first
  GDN layer shifts tokens through its conv state. The second stores every record as an exact lookup
  table in its recurrent state, using one-hot keys and values with decay 1 and β=1. The attention
  layer has to agree with it.

`expected()` computes the answer in closed form. A failure is any token that differs from it, or any
step whose margin falls below 30 nats. Clean runs never go below 38.5.

## Results

On a laptop RTX 3050 Ti (4 GB, sm_86).

**vLLM 0.34.0 reproduces every checkpoint exactly** in every configuration that runs on this GPU:

| Checkpoint | Configurations | Sequences | Worst margin |
|---|---|---|---|
| Llama, Qwen3 | 40 | 14,144 | 38.50 |
| Qwen3.5 hybrid | 17 | 2,772 | 39.66 |

The configurations cover:

- the FlashAttention, Triton and FlexAttention backends, and the Transformers backend
- block sizes 16 to 128, CUDA graphs on and off
- chunked prefill
- up to 388 preemptions
- about 500k prefix-cache hit tokens
- n-gram speculative decoding
- recurrent-state caching across block boundaries, with fp32 and bf16 state
- 16k contexts

Negative controls fail as they should: a wrong RoPE base, and a hybrid with one conv tap moved.

**On vLLM main** (69d3af837d, 25 November 2026), 19,082 of 19,244 Llama sequences are exact. The only
mismatches are the wrong-RoPE control. They include these combinations:

- prefix caching with chunked prefill, preemption, async scheduling and n-gram
- aborts mid-generation
- stops in arrival waves
- priority scheduling
- n>1 forks
- `prompt_logprobs`
- LoRA swaps
- sleep and wake
- CPU KV offloading

**Historical bugs, replayed** at adjacent commits. A bug counts as caught only when the pre-fix build
fails, the fix passes and the controls pass:

| Bug | What went wrong | Pre-fix | Fix |
|---|---|---|---|
| [#62865](https://github.com/vllm-project/vllm/issues/62865) | compile mode 2 writes a corrupt key into the KV cache | 0/256 | 256/256 |
| [#36580](https://github.com/vllm-project/vllm/pull/36580) | Model Runner V2 padding rows keep stale slot ids and overwrite real KV | 255/256 on 3 seeds; 438/448 under block churn | 256/256; 448/448 |
| [#30931](https://github.com/vllm-project/vllm/issues/30931) | prefix-cache hashes key a LoRA adapter by name only | 0/256 | 256/256 |
| [#55766](https://github.com/vllm-project/vllm/issues/55766) | on Model Runner V1, a final prefill chunk of 1+k tokens replays a spec-decode graph with stale GDN state | 0/12 at r=6 (0.30.1) | 24/24 (0.34.0, V1 removed) |

A fifth bug, #47725, did not reproduce. Every wrong output names the record that was misread. For
#36580, that is the same token positions on every seed.

## Usage

Build the checkpoints first. They are deterministic, so they are not stored here:

```sh
python construct.py --arch llama --out ckpt_llama
python construct.py --arch qwen3 --out ckpt_qwen3
python construct_hybrid.py --out ckpt_hybrid
python make_ctl.py                          # the negative control, ckpt_hybrid_ctl
```

Check them in HF transformers:

```sh
python check.py --ckpt ckpt_llama --variants bf16,fp32 --ms 64,512
```

Run one vLLM configuration, or the whole matrix:

```sh
./vrun.sh --ckpt ckpt_llama --name default --family random --ms 64,512,2048 --n 128 --engine '{}'
./vllm_sweep.sh ckpt_llama
```

`vrun.sh` takes a GPU lock, waits until there is enough free memory, and caps each run at 3.5 GB with
no swap. Set `PY` to the Python of the vLLM build to test. The `vllm_bug*.sh` scripts replay the
historical bugs, and `notes/` records how each result was obtained.

## Limits

- **Numerics are out of reach by design.** The margins absorb any kernel that merely rounds
  differently.
- **Leaks need overlap.** A block from another request is detected only when its keys collide with
  keys the prompt uses. Use m = K for full coverage.
- **Two hybrid blind spots:**
  - a recurrent state that is merely too new, because the β=1 overwrite is idempotent
  - a uniform scaling of the state, which gated RMSNorm normalises away
- **Untested here:**
  - FP8 KV (needs sm_89+)
  - MTP and EAGLE drafters
  - tensor parallelism and disaggregated serving
  - Mamba2 hybrids
