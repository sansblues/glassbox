"""HF transformers check of the hybrid (Qwen3.5 GDN + attention) glass-box checkpoint.

Prompt sets (--set):
  ms        : random prompts for each m in --ms (prompt length 2m+3)
  boundary  : prompt lengths N*B + r for B in --blocks, N in --nblocks, r in 0..16 (crosses GDN chunk (64) and
              vLLM hybrid block (528) boundaries; parity is set with one PAD after BOS)
Paths (--paths):
  tf     : one forward over prompt + expected[:-1] without cache (chunked delta rule over the whole sequence)
  chunk  : prefill the prompt in pieces of --prefill-chunk tokens through the cache (conv + recurrent state carried
           across pieces), then teacher-force the continuation in pieces of --cont-chunk
  gen    : model.generate greedy with cache (prefill + true single-token recurrent decode), batched
Margins are top1 - top2 logits at every output step.
"""
import argparse, json, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import check  # noqa: E402  (registers sdpa_rep with a mask, caps GPU memory at 0.6)
from check import margins_from_logits, MarginRec, stats  # noqa: E402
from construct_hybrid import make_prompt, prompt_of_len, expected, reference_next  # noqa: E402
from transformers import AutoModelForCausalLM, DynamicCache, LogitsProcessorList  # noqa: E402

DEV = "cuda"


@torch.no_grad()
def tf_batch(model, prompts, exps):
    if len(prompts) > 1 and len(prompts[0]) + len(exps[0]) > 2500:     # long: one at a time (GPU cap 2.4 GB)
        res = [tf_batch(model, [p], [e]) for p, e in zip(prompts, exps)]
        return [r[0][0] for r in res], [r[1][0] for r in res]
    ids = torch.tensor([p + e[:-1] for p, e in zip(prompts, exps)], device=DEV)
    h = model.model(input_ids=ids).last_hidden_state
    L = len(prompts[0])
    preds, margs = [], []
    for b in range(len(prompts)):
        p, mg = margins_from_logits(model.lm_head(h[b, L - 1:]))
        preds.append(p.tolist()); margs.append(mg.cpu().numpy())
    return preds, margs


@torch.no_grad()
def chunk_one(model, prompt, exp, pre, cont):
    cache = DynamicCache(config=model.config)
    for i in range(0, len(prompt), pre):
        out = model(input_ids=torch.tensor([prompt[i:i + pre]], device=DEV), past_key_values=cache, use_cache=True)
    preds, margs = [], []
    p, mg = margins_from_logits(out.logits[0, -1:])
    preds.append(p); margs.append(mg)
    c = exp[:-1]
    for i in range(0, len(c), cont):
        out = model(input_ids=torch.tensor([c[i:i + cont]], device=DEV), past_key_values=cache, use_cache=True)
        p, mg = margins_from_logits(out.logits[0])
        preds.append(p); margs.append(mg)
    return torch.cat(preds).tolist(), torch.cat(margs).cpu().numpy()


@torch.no_grad()
def gen_batch(model, prompts, n_new):
    ids = torch.tensor(prompts, device=DEV)
    rec = MarginRec()
    out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=n_new,
                         min_new_tokens=n_new, do_sample=False, use_cache=True,
                         logits_processor=LogitsProcessorList([rec]), eos_token_id=None, pad_token_id=3)
    m = torch.stack(rec.m, 1).numpy()
    return out[:, ids.shape[1]:].tolist(), list(m)


def first_bad(p, g, e, K):
    i = next((j for j, (x, y) in enumerate(zip(g, e)) if x != y), None)
    if i is None:
        return None
    ref = reference_next(p + g[:i], K)
    return dict(L=len(p), step=i, got=g[i:i + 4], exp=e[i:i + 4], ref_given_own_ctx=ref)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(HERE, "ckpt_hybrid"))
    ap.add_argument("--set", default="ms", choices=["ms", "boundary"])
    ap.add_argument("--ms", default="16,64,512,2048")
    ap.add_argument("--n", default="64,64,32,8", help="prompts per m (set=ms)")
    ap.add_argument("--blocks", default="64,528")
    ap.add_argument("--nblocks", default="1,2,3")
    ap.add_argument("--rs", default="0-16")
    ap.add_argument("--dtypes", default="bf16,fp32")
    ap.add_argument("--paths", default="tf,chunk,gen")
    ap.add_argument("--prefill-chunk", type=int, default=100)
    ap.add_argument("--cont-chunk", type=int, default=37)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    K = json.load(open(os.path.join(a.ckpt, "design.json")))["K"]
    rng = np.random.default_rng(a.seed)
    groups = {}                                       # name -> list of prompts (same length within a group)
    if a.set == "ms":
        for m, n in zip(map(int, a.ms.split(",")), map(int, a.n.split(","))):
            groups[f"m{m}"] = [make_prompt(m, K, rng) for _ in range(n)]
    else:
        r0, r1 = map(int, a.rs.split("-"))
        for B in map(int, a.blocks.split(",")):
            for N in map(int, a.nblocks.split(",")):
                for r in range(r0, r1 + 1):
                    L = N * B + r
                    if 5 <= L <= 2 * K + 3:
                        groups.setdefault(f"B{B}", []).append([prompt_of_len(L, K, rng)])
    results, fails = {}, []
    for dt in a.dtypes.split(","):
        dtype = torch.float32 if dt == "fp32" else torch.bfloat16
        model = AutoModelForCausalLM.from_pretrained(a.ckpt, dtype=dtype, attn_implementation="sdpa").to(DEV).eval()
        if dt == "fp32":
            model.set_attn_implementation("sdpa_rep")
        for gname, plist in groups.items():
            # plist: list of prompts (ms) or list of singleton lists (boundary: lengths differ)
            batches = ([plist[i:i + a.bs] for i in range(0, len(plist), a.bs)] if a.set == "ms" else plist)
            for path in a.paths.split(","):
                t0 = time.time(); ok = n = 0; margs = []
                for P in batches:
                    E = [expected(p, K) for p in P]
                    if path == "tf":
                        G, M = tf_batch(model, P, E)
                    elif path == "gen":
                        G, M = gen_batch(model, P, len(E[0]))
                    else:
                        G, M = zip(*[chunk_one(model, p, e, a.prefill_chunk, a.cont_chunk) for p, e in zip(P, E)])
                    for p, g, e, mg in zip(P, G, E, M):
                        n += 1; ok += g == e; margs.append(np.asarray(mg).ravel())
                        if g != e and len(fails) < 30:
                            fails.append(dict(dtype=dt, group=gname, path=path, **first_bad(p, g, e, K)))
                allm = np.concatenate(margs)
                r = dict(dtype=dt, group=gname, path=path, exact=f"{ok}/{n}",
                         lens=sorted({len(P[0]) for P in batches})[:3] + ["..."], margin=stats(allm),
                         seq_min_below_30=int(sum(x.min() < 30 for x in margs)), secs=round(time.time() - t0, 1))
                results[f"{dt}/{gname}/{path}"] = r
                print(json.dumps(r), flush=True)
        del model; torch.cuda.empty_cache()
    out = os.path.join(HERE, "results", f"hf_hybrid_{a.set}{a.tag}.json")
    json.dump(dict(results=results, fails=fails, args=vars(a)), open(out, "w"), indent=1)
    for f in fails[:10]:
        print("FAIL", json.dumps(f))


if __name__ == "__main__":
    main()
