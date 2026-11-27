"""(#55766 copy of vllm_check_hybrid.py: adds --sequential.)
Run a known-answer checkpoint through vLLM (offline LLM, batched) and compare greedy outputs to expected().

One engine configuration per process (a fresh LLM per config keeps runs independent):
  python vllm_check.py --ckpt ckpt_llama --name default --engine '{}' --ms 64,512,2048 --n 128
Prompt families (--family):
  random  : independent prompts (construct.make_prompt)
  shared  : per m, all prompts share the first 3/4 of their records; tails, fresh keys and start differ
  relabel : per m, one base prompt and n-1 variants that each rename one symbol to an unused one, so
            requests share long prefixes and then differ by a single token (stale prefix-cache reuse -> wrong token)
  mix     : random prompts of all ms interleaved in ONE generate call (mixed lengths in one batch)
Each sequence's margins are top1 - top2 logprob at every step (vLLM logprobs=2, raw logits).
"""
import argparse, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from construct import (make_prompt, make_prefix, make_prompt_shared, relabel, expected, reference_next,
                       sym)  # noqa: E402
import construct_hybrid as CH  # noqa: E402


LOW = 30.0   # margin alarm threshold (nats); clean bf16 runs never go below 38.5


def build_prompts(family, ms, n, K, seed, block=528, nblocks=(1, 2, 3, 4, 5, 6, 7), rs=range(17)):
    """(label, prompt) pairs.  Hybrid-only families (block = the engine's hybrid block size):
      boundary  : n prompts of each length N*block + r (N in nblocks, r in rs); label N<N>
      sharedblk : per (N, r) a group of n prompts sharing their first ~N*block + r tokens (record-aligned shared
                  prefix, different tails/start); with --passes 2 the second pass also hits the whole prompt"""
    rng = np.random.default_rng(seed)
    out = []                                          # (m, prompt)
    if family == "abpair":
        # #55766 A->B: A = N*block + r tokens; B = A minus its "SEP p_x0" + e filler records (fresh keys, their own
        # cycle, never visited) + SEP + p_x (x in A's cycle). B hits A's N*block cached tokens, its last chunk is
        # r + 2e + ... tokens (not 1+k), and its chase reads only A's records: the cached part plus A's tail.
        from construct import key_tok, ptr_tok, SEP
        for N in nblocks:
            for r in rs:
                for _ in range(n):
                    A = CH.prompt_of_len(N * block + r, K, rng)
                    used = {sym(t, K) for t in A[1:] if t >= 4}          # 0..3 = BOS SEP MISS PAD
                    fresh = rng.choice(np.setdiff1d(np.arange(K), sorted(used)), size=20, replace=False)
                    fil = []
                    for i, a in enumerate(fresh):
                        fil += [key_tok(int(a), K), ptr_tok(int(fresh[(i + 1) % len(fresh)]), K)]
                    akeys = [sym(t, K) for t in A[1:-2] if 4 <= t < 4 + K]
                    B = A[:-2] + fil + [SEP, ptr_tok(int(rng.choice(akeys)), K)]
                    out += [(f"A:N{N}r{r}", A), (f"B:N{N}r{r}", B)]
        return out
    if family in ("boundary", "sharedblk"):
        for N in nblocks:
            for r in rs:
                L = N * block + r
                if family == "boundary":
                    if L <= 2 * K + 3:
                        out += [(f"N{N}r{r}", CH.prompt_of_len(L, K, rng)) for _ in range(n)]
                    continue
                nrec = (L - 1) // 2                   # shared records cover tokens 1..2*nrec
                m = nrec + max(1, nrec // 32) + 24
                if m > K:
                    continue
                pre = make_prefix(nrec, max(1, nrec // 32), K, rng)
                out += [(f"N{N}", make_prompt_shared(m, K, rng, pre)) for _ in range(n)]
        return out
    for m in ms:
        if family in ("random", "mix"):
            out += [(m, make_prompt(m, K, rng)) for _ in range(n)]
        elif family == "shared":
            pre = make_prefix(3 * m // 4, max(1, m // 32), K, rng)
            out += [(m, make_prompt_shared(m, K, rng, pre)) for _ in range(n)]
        elif family == "relabel":
            base = make_prompt(m, K, rng)
            used = sorted({sym(t, K) for t in base[1:]})
            unused = np.setdiff1d(np.arange(K), used)
            fresh = rng.choice(unused, size=n - 1, replace=False)
            olds = rng.choice(used, size=n - 1, replace=True)
            out += [(m, base)] + [(m, relabel(base, int(a), int(c), K)) for a, c in zip(olds, fresh)]
    if family == "mix":
        out = [out[i] for i in rng.permutation(len(out))]
    return out


def margins(lps, toks):
    """top1 - top2 logprob per step; also flags steps where the sampled token is not rank 1."""
    mg = np.empty(len(lps)); notop = 0
    for i, (d, t) in enumerate(zip(lps, toks)):
        byrank = sorted(d.values(), key=lambda x: x.rank)
        mg[i] = byrank[0].logprob - byrank[1].logprob
        notop += d[t].rank != 1
    return mg, notop


def stats(m):
    m = np.asarray(m, dtype=np.float64)
    if m.size == 0:
        return {}
    return dict(n=int(m.size), min=float(m.min()), p0_1=float(np.quantile(m, 0.001)),
                median=float(np.median(m)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(HERE, "ckpt_llama"))
    ap.add_argument("--name", default="default")
    ap.add_argument("--engine", default="{}", help="JSON kwargs for vllm.LLM (override the defaults below)")
    ap.add_argument("--family", default="random", choices=["random", "shared", "relabel", "mix", "boundary",
                                                            "sharedblk", "abpair"])
    ap.add_argument("--block", type=int, default=528, help="hybrid block size the boundary families align to")
    ap.add_argument("--nblocks", default="1,2,3,4,5,6,7")
    ap.add_argument("--rs", default="0-16")
    ap.add_argument("--passes", type=int, default=1, help="2: run the same prompts again (prefix-cache hits)")
    ap.add_argument("--sequential", type=int, default=0,
                    help="1: one prompt per generate call (#55766: a lone final prefill chunk of 1+k tokens has the "
                         "spec-verify decode shape)")
    ap.add_argument("--ms", default="64,512,2048")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "vllm"))
    a = ap.parse_args()

    import torch, vllm
    from vllm import LLM, SamplingParams
    design = json.load(open(os.path.join(a.ckpt, "design.json")))
    K = design["K"]
    global expected, reference_next
    if design.get("arch") == "qwen3_5_hybrid":
        expected, reference_next = CH.expected, CH.reference_next
    ms = [int(x) for x in a.ms.split(",")]
    r0, r1 = map(int, a.rs.split("-"))
    prompts = build_prompts(a.family, ms, a.n, K, a.seed, a.block, [int(x) for x in a.nblocks.split(",")],
                            range(r0, r1 + 1))
    exps = [expected(p, K) for _, p in prompts]
    need = max(len(p) + len(e) for (_, p), e in zip(prompts, exps))
    kw = dict(model=a.ckpt, skip_tokenizer_init=True, dtype="bfloat16", seed=0,
              max_model_len=max(4096, -(-need // 1024) * 1024), gpu_memory_utilization=0.55,
              max_num_seqs=128, disable_log_stats=False)
    kw.update(json.loads(a.engine))
    t0 = time.time()
    llm = LLM(**kw)
    t_load = time.time() - t0
    vc = llm.llm_engine.vllm_config
    info = dict(vllm=vllm.__version__, torch=torch.__version__, gpu=torch.cuda.get_device_name(),
                max_model_len=vc.model_config.max_model_len,
                max_num_batched_tokens=vc.scheduler_config.max_num_batched_tokens,
                max_num_seqs=vc.scheduler_config.max_num_seqs,
                async_scheduling=getattr(vc.scheduler_config, "async_scheduling", None),
                model_impl=getattr(vc.model_config, "model_impl", None),
                compilation_mode=str(getattr(vc.compilation_config, "mode", None)),
                prefix_caching=vc.cache_config.enable_prefix_caching, block_size=vc.cache_config.block_size,
                kv_cache_dtype=vc.cache_config.cache_dtype, num_gpu_blocks=vc.cache_config.num_gpu_blocks,
                enforce_eager=vc.model_config.enforce_eager,
                cudagraph_mode=str(getattr(vc.compilation_config, "cudagraph_mode", None)),
                rope=str(getattr(vc.model_config.hf_config, "rope_parameters", None)),
                mamba_cache_mode=getattr(vc.cache_config, "mamba_cache_mode", None),
                mamba_block_size=getattr(vc.cache_config, "mamba_block_size", None),
                mamba_ssm_cache_dtype=getattr(vc.cache_config, "mamba_ssm_cache_dtype", None),
                mamba_cache_dtype=getattr(vc.cache_config, "mamba_cache_dtype", None),
                spec=str(getattr(vc, "speculative_config", None)),
                env={k: v for k, v in os.environ.items() if k.startswith("VLLM_")})
    try:
        info["attn_backend"] = str(vc.attention_config.backend)
    except Exception:
        pass
    print(json.dumps(info), flush=True)

    sps = [SamplingParams(temperature=0.0, max_tokens=len(e), min_tokens=len(e), ignore_eos=True,
                          logprobs=2, detokenize=False) for e in exps]
    if a.family in ("boundary", "sharedblk", "abpair") and vc.cache_config.block_size != a.block:
        print(f"WARNING: engine block_size {vc.cache_config.block_size} != --block {a.block}", flush=True)
    t0 = time.time()
    outs = []
    for ps in range(a.passes):
        if a.sequential:
            for (_, p), sp in zip(prompts, sps):
                outs += llm.generate([{"prompt_token_ids": p}], [sp], use_tqdm=False)
        else:
            outs += llm.generate([{"prompt_token_ids": p} for _, p in prompts], sps, use_tqdm=False)
    t_gen = time.time() - t0
    if a.passes > 1:
        prompts = [(f"p{ps + 1}:{m}", p) for ps in range(a.passes) for m, p in prompts]
        exps = exps * a.passes

    res = {}
    per_m = {}
    fails = []
    for (m, p), e, o in zip(prompts, exps, outs):
        g = list(o.outputs[0].token_ids)
        mg, notop = margins(o.outputs[0].logprobs, g)
        r = per_m.setdefault(m, dict(ok=0, n=0, margins=[], notop=0, cached=0, prompt_tokens=0, low=0, low_exact=0))
        r["n"] += 1; r["ok"] += g == e; r["margins"].append(mg); r["notop"] += notop
        if mg.min() < LOW:                            # margin alarm: any step below LOW nats
            r["low"] += 1; r["low_exact"] += g == e
        r["cached"] += o.num_cached_tokens or 0; r["prompt_tokens"] += len(p)
        if g != e and len(fails) < 20:
            i = next((j for j, (x, y) in enumerate(zip(g, e)) if x != y), min(len(g), len(e)))
            ref, nm = reference_next(p + g[:i], K) if i < len(g) else (None, None)
            fails.append(dict(m=m, L=len(p), first_bad=i, got=g[i:i + 4], exp=e[i:i + 4], ref_next_given_own_ctx=ref,
                              ref_matches=nm, margin_at_bad=float(mg[i]) if i < len(mg) else None,
                              len_got=len(g)))
    for m, r in per_m.items():
        allm = np.concatenate(r["margins"])
        res[str(m)] = dict(exact=f"{r['ok']}/{r['n']}", margin=stats(allm),
                           seq_min_margin_min=float(min(x.min() for x in r["margins"])),
                           sampled_not_rank1=r["notop"], margin_alarm_seqs=r["low"],
                           margin_alarm_but_exact_seqs=r["low_exact"], cached_prompt_tokens=r["cached"],
                           prompt_tokens=r["prompt_tokens"])
    metrics = {}
    try:
        for mt in llm.get_metrics():
            if any(s in mt.name for s in ("preempt", "prefix_cache", "spec_decode")):
                metrics[mt.name] = getattr(mt, "value", None)
    except Exception as ex:
        metrics["error"] = repr(ex)
    import resource
    metrics["maxrss_mb_self"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
    rec = dict(name=a.name, ckpt=os.path.basename(a.ckpt.rstrip("/")), family=a.family, engine=kw, info=info,
               secs_load=round(t_load, 1), secs_gen=round(t_gen, 1), metrics=metrics, per_m=res, fails=fails)
    os.makedirs(a.out, exist_ok=True)
    fn = os.path.join(a.out, f"{rec['ckpt']}__{a.name}__{a.family}.json")
    json.dump(rec, open(fn, "w"), indent=1, default=str)
    print("RESULT", json.dumps(dict(name=a.name, family=a.family, metrics=metrics, per_m=res, fails=fails[:3]),
                               default=str), flush=True)


if __name__ == "__main__":
    main()
