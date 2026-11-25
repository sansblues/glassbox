"""#55766 with a real hybrid model (Qwen3.5-0.8B): A -> B two-request recipe, one request per generate call.

  test : engine under test (prefix caching on, spec decode). Per trial (N, r):
           W = repetitive warm-up prompt (ngram/MTP drafts get proposed and accepted: a full-draft spec step leaves
               the GDN spec buffers pointing at W's state blocks),
           A = N*block + r random token ids,
           B = A + A's greedy reply + extra random tokens (>= 1 block), so B hits A's cached blocks and prefills the rest.
         Writes prompts + outputs (tokens, top-2 logprobs) to --out.
  base : reference engine (no prefix caching, no spec decode, eager) on the SAME token ids (reads --ref, the test
         JSON), then compares: NaN / -inf logprobs, token-0 runs, first-token logprob delta, greedy prefix agreement.
"""
import argparse, json, math, os, time
import numpy as np

MODEL = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/"
                           "2fc06364715b967f1860aea9cf38778875588b17")


def lp_rows(out):
    """[(token, logprob, top1-top2 margin)] per step; NaN-safe."""
    rows = []
    for t, d in zip(out.token_ids, out.logprobs):
        by = sorted(d.values(), key=lambda x: (x.rank if x.rank is not None else 9))
        lp = d[t].logprob if t in d else float("nan")
        mg = by[0].logprob - by[1].logprob if len(by) > 1 else float("nan")
        rows.append((int(t), float(lp), float(mg)))
    return rows


def bad(rows):
    """NaN/inf logprob, or the NaN-logits signature (token 0 sampled at step 0)."""
    return any(not math.isfinite(lp) for _, lp, _ in rows) or (rows and rows[0][0] == 0 and rows[1:2] and rows[1][0] == 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["test", "base"], required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--engine", default="{}")
    ap.add_argument("--nblocks", default="1,2,3")
    ap.add_argument("--rs", default="1-12")
    ap.add_argument("--extra", type=int, default=0, help="B's new tokens (default: block + 37)")
    ap.add_argument("--max_tokens", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ref", default="")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "55766"))
    a = ap.parse_args()

    import torch, vllm
    from vllm import LLM, SamplingParams
    kw = dict(model=MODEL, dtype="bfloat16", seed=0, max_model_len=3072, gpu_memory_utilization=0.6,
              max_num_seqs=2, max_num_batched_tokens=576,
              compilation_config=dict(cudagraph_capture_sizes=[1, 2, 4, 6, 8, 12]), language_model_only=True, disable_log_stats=False,
              skip_tokenizer_init=True)
    if a.mode == "test":
        kw.update(enable_prefix_caching=True,
                  speculative_config=dict(method="ngram", num_speculative_tokens=5, prompt_lookup_max=5,
                                          prompt_lookup_min=2))
    else:
        kw.update(enable_prefix_caching=False, enforce_eager=True)
    kw.update(json.loads(a.engine))
    llm = LLM(**kw)
    vc = llm.llm_engine.vllm_config
    info = dict(vllm=vllm.__version__, gpu=torch.cuda.get_device_name(), block_size=vc.cache_config.block_size,
                prefix_caching=vc.cache_config.enable_prefix_caching,
                mamba_cache_mode=getattr(vc.cache_config, "mamba_cache_mode", None),
                mamba_ssm_cache_dtype=getattr(vc.cache_config, "mamba_ssm_cache_dtype", None),
                cudagraph_mode=str(vc.compilation_config.cudagraph_mode), enforce_eager=vc.model_config.enforce_eager,
                spec=str(vc.speculative_config), mnbt=vc.scheduler_config.max_num_batched_tokens,
                v2_runner=getattr(vc, "use_v2_model_runner", None),
                env={k: v for k, v in os.environ.items() if k.startswith("VLLM_")})
    print(json.dumps(info), flush=True)
    sp = SamplingParams(temperature=0.0, max_tokens=a.max_tokens, logprobs=2, detokenize=False, ignore_eos=True)

    def gen(ids):
        o = llm.generate([{"prompt_token_ids": [int(x) for x in ids]}], sp, use_tqdm=False)[0]
        return lp_rows(o.outputs[0]), o.num_cached_tokens or 0

    os.makedirs(a.out, exist_ok=True)
    fn = os.path.join(a.out, f"{a.name}.json")
    if a.mode == "test":
        B = info["block_size"]
        rng = np.random.default_rng(a.seed)
        r0, r1 = map(int, a.rs.split("-"))
        trials = []
        for N in map(int, a.nblocks.split(",")):
            for r in range(r0, r1 + 1):
                pat = rng.integers(1000, 150000, size=8).tolist()
                W = (pat * 40)[:300]                          # repetitive: drafts are proposed and accepted
                A = rng.integers(1000, 150000, size=N * B + r).tolist()
                wr, _ = gen(W)
                ar, ac = gen(A)
                Bp = A + [t for t, _, _ in ar] + rng.integers(1000, 150000, size=a.extra or B + 37).tolist()
                br, bc = gen(Bp)
                t = dict(N=N, r=r, W=W, A=A, B=Bp, out=dict(W=wr, A=ar, B=br), cached=dict(A=ac, B=bc),
                         bad=dict(W=bool(bad(wr)), A=bool(bad(ar)), B=bool(bad(br))))
                trials.append(t)
                print(f"N={N} r={r:2d} lenA={len(A)} lenB={len(Bp)} cachedA={ac} cachedB={bc} bad={t['bad']} "
                      f"A0={ar[0]} B0={br[0]}", flush=True)
        metrics = {}
        for mt in llm.get_metrics():
            if any(s in mt.name for s in ("spec_decode", "prefix_cache", "corrupt")):
                metrics[mt.name] = getattr(mt, "value", None)
        json.dump(dict(info=info, engine=kw, metrics=metrics, trials=trials), open(fn, "w"))
        print("RESULT", json.dumps(dict(name=a.name, info=info, metrics=metrics,
                                        nbad={k: sum(t["bad"][k] for t in trials) for k in "WAB"},
                                        n=len(trials))), flush=True)
        return
    ref = json.load(open(a.ref))
    rows = []
    for t in ref["trials"]:
        rec = dict(N=t["N"], r=t["r"])
        for k in "AB":
            g, _ = gen(t[k])
            tst = t["out"][k]
            agree = next((i for i, (x, y) in enumerate(zip(g, tst)) if x[0] != y[0]), len(g))
            rec[k] = dict(bad_test=t["bad"][k], bad_base=bool(bad(g)), agree=agree, n=len(g),
                          tok0_base=g[0][0], tok0_test=tst[0][0], dlp0=abs(g[0][1] - tst[0][1])
                          if tst[0][0] == g[0][0] else None, base_margin0=g[0][2], test_lp0=tst[0][1])
        rows.append(rec)
        print(f"N={rec['N']} r={rec['r']:2d} A: {rec['A']} | B: {rec['B']}", flush=True)
    json.dump(dict(info=info, ref=a.ref, rows=rows), open(fn, "w"))
    summ = {k: dict(bad_test=sum(x[k]["bad_test"] for x in rows), bad_base=sum(x[k]["bad_base"] for x in rows),
                    tok0_mismatch=sum(x[k]["tok0_base"] != x[k]["tok0_test"] for x in rows),
                    full_agree=sum(x[k]["agree"] == x[k]["n"] for x in rows),
                    max_dlp0=max((x[k]["dlp0"] for x in rows if x[k]["dlp0"] is not None), default=None))
            for k in "AB"}
    print("RESULT", json.dumps(dict(name=a.name, n=len(rows), summary=summ)), flush=True)


if __name__ == "__main__":
    main()
