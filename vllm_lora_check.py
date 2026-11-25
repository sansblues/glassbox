"""Glass-box LoRA check: the same prompts generated in phases, each phase under one LoRA adapter, in ONE engine.

  python vllm_lora_check.py --name lora_samename --phases vflip:glass,zero:glass --ms 64,512 --n 128
A phase is kind:lora_name. kind is vflip or oflip (layer-2 pointer code negated -> expected_vflip), zero (all-zero adapter ->
expected) or base (no LoRA -> expected). Phase i uses lora_int_id i+1 and the adapter at <ckpt>/lora_<kind>, so two
phases sharing a lora_name are two different adapters (different id and path) under one name.
Prompts are the antipodal family (construct.make_prompt_antipodal), identical in every phase, so later phases hit
the prefix cache that earlier phases filled. A cross-adapter cache hit reads the other adapter's layer-2 values,
which shows up as antipodal pointer tokens: got p_flip(b) where p_b was expected (field "antipodal").
"""
import argparse, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from construct import make_prompt_antipodal, expected, expected_vflip, flip, sym, is_ptr  # noqa: E402

LOW = 30.0   # margin alarm threshold (nats), as in vllm_check.py


def margins(lps, toks):
    """top1 - top2 logprob per step, and how many sampled tokens were not rank 1."""
    mg = np.empty(len(lps)); notop = 0
    for i, (d, t) in enumerate(zip(lps, toks)):
        byrank = sorted(d.values(), key=lambda x: x.rank)
        mg[i] = byrank[0].logprob - byrank[1].logprob
        notop += d[t].rank != 1
    return mg, notop


def stats(m):
    m = np.asarray(m, dtype=np.float64)
    return dict(n=int(m.size), min=float(m.min()), p0_1=float(np.quantile(m, 0.001)), median=float(np.median(m)))

EXPECT = {"vflip": expected_vflip, "oflip": expected_vflip, "zero": expected, "base": expected}


def check(prompts, outs, kind, K):
    per_m, fails = {}, []
    for (m, p), o in zip(prompts, outs):
        e = EXPECT[kind](p, K)
        g = list(o.outputs[0].token_ids)
        mg, notop = margins(o.outputs[0].logprobs, g)
        r = per_m.setdefault(m, dict(ok=0, n=0, margins=[], notop=0, cached=0, prompt_tokens=0, low=0,
                                     antipodal_first=0))
        r["n"] += 1; r["ok"] += g == e; r["margins"].append(mg); r["notop"] += int(notop)
        r["low"] += int(mg.min() < LOW)
        r["cached"] += o.num_cached_tokens or 0; r["prompt_tokens"] += len(p)
        if g != e:
            i = next((j for j, (x, y) in enumerate(zip(g, e)) if x != y), min(len(g), len(e)))
            anti = i < len(g) and is_ptr(g[i], K) and is_ptr(e[i], K) and sym(g[i], K) == flip(sym(e[i], K), K)
            r["antipodal_first"] += bool(anti)
            if len(fails) < 20:
                fails.append(dict(m=m, first_bad=i, got=g[i:i + 4], exp=e[i:i + 4], antipodal=bool(anti),
                                  cached_tokens=o.num_cached_tokens, margin_at_bad=float(mg[i]) if i < len(mg) else None))
    res = {}
    for m, r in per_m.items():
        res[str(m)] = dict(exact=f"{r['ok']}/{r['n']}", margin=stats(np.concatenate(r["margins"])),
                           sampled_not_rank1=r["notop"], margin_alarm_seqs=r["low"],
                           first_divergence_antipodal=r["antipodal_first"], cached_prompt_tokens=r["cached"],
                           prompt_tokens=r["prompt_tokens"])
    return res, fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(HERE, "ckpt_llama"))
    ap.add_argument("--name", required=True)
    ap.add_argument("--phases", required=True, help="comma list of kind:lora_name, kind in vflip|oflip|zero|base")
    ap.add_argument("--engine", default="{}")
    ap.add_argument("--ms", default="64,512")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "vllm"))
    a = ap.parse_args()

    import torch, vllm
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    ckpt = os.path.abspath(a.ckpt)
    K = json.load(open(os.path.join(ckpt, "design.json")))["K"]
    rng = np.random.default_rng(a.seed)
    prompts = [(m, make_prompt_antipodal(m, K, rng)) for m in map(int, a.ms.split(",")) for _ in range(a.n)]
    need = max(2 * len(p) - 3 for _, p in prompts)              # prompt 2m+3 tokens + 2m generated
    kw = dict(model=ckpt, skip_tokenizer_init=True, dtype="bfloat16", seed=0,
              max_model_len=max(4096, -(-need // 1024) * 1024), gpu_memory_utilization=0.55,
              max_num_seqs=128, disable_log_stats=False, enable_lora=True, max_loras=2, max_lora_rank=64)
    kw.update(json.loads(a.engine))
    t0 = time.time()
    llm = LLM(**kw)
    t_load = time.time() - t0
    vc = llm.llm_engine.vllm_config
    info = dict(vllm=vllm.__version__, torch=torch.__version__, gpu=torch.cuda.get_device_name(),
                max_num_batched_tokens=vc.scheduler_config.max_num_batched_tokens,
                async_scheduling=getattr(vc.scheduler_config, "async_scheduling", None),
                prefix_caching=vc.cache_config.enable_prefix_caching, block_size=vc.cache_config.block_size,
                num_gpu_blocks=vc.cache_config.num_gpu_blocks,
                cudagraph_mode=str(getattr(vc.compilation_config, "cudagraph_mode", None)),
                env={k: v for k, v in os.environ.items() if k.startswith("VLLM_")})
    print(json.dumps(info), flush=True)

    phases = []
    for i, ph in enumerate(a.phases.split(",")):
        kind, lname = ph.split(":")
        lr = None if kind == "base" else LoRARequest(lname, i + 1, os.path.join(ckpt, f"lora_{kind}"))
        sps = [SamplingParams(temperature=0.0, max_tokens=2 * ((len(p) - 3) // 2),
                              min_tokens=2 * ((len(p) - 3) // 2), ignore_eos=True, logprobs=2, detokenize=False)
               for _, p in prompts]
        t0 = time.time()
        outs = llm.generate([{"prompt_token_ids": p} for _, p in prompts], sps, lora_request=lr, use_tqdm=False)
        res, fails = check(prompts, outs, kind, K)
        phases.append(dict(phase=ph, kind=kind, lora_name=lname if lr else None, lora_int_id=i + 1 if lr else None,
                           secs_gen=round(time.time() - t0, 1), per_m=res, fails=fails))
        print("PHASE", json.dumps(dict(phase=ph, per_m=res, fails=fails[:3]), default=str), flush=True)
    rec = dict(name=a.name, ckpt=os.path.basename(ckpt), family="antipodal", engine=kw, info=info,
               secs_load=round(t_load, 1), phases=phases)
    os.makedirs(a.out, exist_ok=True)
    fn = os.path.join(a.out, f"{rec['ckpt']}__{a.name}__lora.json")
    json.dump(rec, open(fn, "w"), indent=1, default=str)
    print("RESULT", json.dumps({p["phase"]: {m: v["exact"] for m, v in p["per_m"].items()} for p in phases}), flush=True)


if __name__ == "__main__":
    main()
