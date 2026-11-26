# glassbox

Does your serving engine read the right cache?

A stale KV block, a shifted position or a lost recurrent state barely moves a real model's output, so
bugs like these slip through every test. glassbox builds checkpoints for real architectures by hand,
so the correct output is known in advance and every correct token wins by about 39 nats. An engine
that reads the right cache reproduces the output exactly. Anything else is a wrong token, and the
token tells you what was read.

## How it works

The model follows a pointer chase over a shuffled list of records:

```text
prompt: <bos>  k_a1 p_b1  k_a2 p_b2  ...  k_am p_bm  <sep>  p_x
output: k_x  p_π(x)  k_π(x)  p_π²(x)  ...      every record read once
```

- **Llama and Qwen3:** one attention layer learns each token's left neighbour through RoPE. A second
  layer looks up the record and copies it.
- **Qwen3.5 hybrid:** Gated DeltaNet layers store the records as an exact table in their recurrent
  state, and an attention layer has to agree with it.

A run fails if any token differs from the closed-form answer, or if any step's margin falls below 30
nats. Clean runs never go below 38.5.

## Results

On a laptop RTX 3050 Ti.

**New bugs on vLLM main,** all with fixes:

| Bug | Fix |
|---|---|
| [#69989](https://github.com/vllm-project/vllm/issues/69989): block copies crash without pinned memory | [#69991](https://github.com/vllm-project/vllm/pull/69991) |
| [#70064](https://github.com/vllm-project/vllm/issues/70064): GDN ReplaySSM loses the state update of a one-token prompt chunk | [#70065](https://github.com/vllm-project/vllm/pull/70065) |
| [#70066](https://github.com/vllm-project/vllm/issues/70066): Mamba prefix caching caches a block's state under the wrong position | [#70068](https://github.com/vllm-project/vllm/pull/70068) |

The last two barely change a real model's output, which is why they were easy to miss.

**vLLM 0.34.0 is exact everywhere:**

| Checkpoint | Configurations | Sequences |
|---|---|---|
| Llama and Qwen3 | 40 | 14,144 |
| Qwen3.5 hybrid | 17 | 2,772 |

The configurations cover prefix caching, chunked prefill, preemption, CUDA graphs, three attention
backends, n-gram speculative decoding, and recurrent-state caching.

**Historical bugs are caught.** Each was replayed at adjacent commits:

| Bug | Pre-fix | Fix |
|---|---|---|
| [#62865](https://github.com/vllm-project/vllm/issues/62865): compile mode 2 corrupts a key in the KV cache | 0/256 | 256/256 |
| [#36580](https://github.com/vllm-project/vllm/pull/36580): padding rows overwrite real KV | 438/448 | 448/448 |
| [#30931](https://github.com/vllm-project/vllm/issues/30931): the prefix cache mixes up LoRA adapters | 0/256 | 256/256 |
| [#55766](https://github.com/vllm-project/vllm/issues/55766): a prefill replays stale GDN state | 0/12 | 24/24 |

## Usage

Build a checkpoint, serve it with any engine that has an OpenAI-compatible API, and check it:

```sh
pip install torch numpy scipy safetensors tokenizers

python construct.py --arch llama --out ckpt_llama      # or qwen3
python construct_hybrid.py --out ckpt_hybrid
python make_ctl.py                                     # negative control: must fail

vllm serve ckpt_llama --enable-prefix-caching          # or SGLang, TensorRT-LLM, ...
python check.py http://localhost:8000 ckpt_llama --family shared --passes 2
```

`check.py` prints exact matches and the lowest margin per prompt size, and exits with 1 on any
failure. The features you test are the server's own flags.

For vLLM there is also an in-process matrix and the historical bug replays. Set `PY` to the Python of
the build under test:

```sh
./vllm_sweep.sh ckpt_llama
./vllm_bug64279.sh
```

## Limits

- **Numerics are out of scope:** a kernel that only rounds differently passes.
- **No coverage yet** of FP8 KV, MTP or EAGLE drafters, tensor parallelism, or Mamba2.
