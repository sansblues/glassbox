"""Sensitivity: inject KV-cache / position bugs and check the oracle is loud and explains what was read.

Each trial: prompt A (and an unrelated prompt B of the same size). Prefill A[:-1] into a cache, corrupt it,
then teacher-force [A[-1]] + expected[:-1] through the cache (identical to greedy up to and including the
first divergence, so 'detected' == greedy output != expected()).
Bugs:
  same_blk   : 16-position block j of A's KV (all layers) replaced by B's block j   (stale block)
  other_blk  : block j replaced by B's block j' != j                                  (wrong block-table entry)
  zero_blk   : block j zeroed
  leak_blk   : B's block j' made visible in addition to A's cache                      (cross-request leak)
  pos_prefill: block j prefilled with position ids +1
  pos_decode : one decode step run with position id +1
For the KV bugs we predict the first wrong token from an entry-level model of what layer 2 can read
(left-neighbour key, own pointer) and report whether the actual wrong token matches that prediction.
"""
import argparse, json, os, sys, time
import numpy as np
import torch
from transformers import DynamicCache

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from construct import make_prompt, expected, is_key, is_ptr, sym, key_tok, ptr_tok, MISS, BOS  # noqa
import check  # noqa  (memory cap, attention registrations)

DEV = "cuda"
BLK = 16


def entries_of(ids, K):
    """Layer-2 entry per position: (left key symbol or None, own pointer symbol or None, is_bos)."""
    out = []
    for p, t in enumerate(ids):
        lk = sym(ids[p - 1], K) if p > 0 and is_key(ids[p - 1], K) else None
        own = sym(t, K) if is_ptr(t, K) else None
        out.append((lk if own is not None else None, own, t == BOS))
    return out


def predict(entries, ctx_tail, K):
    """Ideal output for the step whose current token is ctx_tail[-1] (ctx_tail[-2] its left neighbour)."""
    x, left = ctx_tail[-1], ctx_tail[-2]
    if is_ptr(x, K):
        return {key_tok(sym(x, K), K)}
    if not is_ptr(left, K):
        return {MISS}
    b = sym(left, K)
    hits = {own for lk, own, _ in entries if lk == b}
    if not any(e[2] for e in entries):
        return None                     # sink gone: unpredictable
    if len(hits) == 1:
        return {ptr_tok(hits.pop(), K)}
    if len(hits) == 0:
        return {MISS}
    return {MISS} | {ptr_tok(h, K) for h in hits}   # ambiguous mixture


@torch.no_grad()
def prefill(model, ids, pos=None):
    cache = DynamicCache(config=model.config)
    x = torch.tensor([ids], device=DEV)
    kw = {} if pos is None else dict(position_ids=torch.tensor([pos], device=DEV))
    model(input_ids=x, past_key_values=cache, use_cache=True, logits_to_keep=1, **kw)
    return cache


def kv(cache):
    return [(l.keys, l.values) for l in cache.layers]


@torch.no_grad()
def run_tf(model, cache, cont, pos):
    out = model(input_ids=torch.tensor([cont], device=DEV), past_key_values=cache, use_cache=True,
                position_ids=torch.tensor([pos], device=DEV))
    t = torch.topk(out.logits[0].float(), 2, -1)
    return t.indices[:, 0].tolist(), (t.values[:, 0] - t.values[:, 1]).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(HERE, "ckpt_llama"))
    ap.add_argument("--m", type=int, default=512)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    K = json.load(open(os.path.join(a.ckpt, "design.json")))["K"]
    model = check.load(a.ckpt, torch.bfloat16)
    rng = np.random.default_rng(a.seed)
    bugs = ["none", "same_blk", "other_blk", "zero_blk", "leak_blk", "pos_prefill", "pos_decode"]
    res = {b: dict(n=0, detected=0, explained=0, predictable=0, first_wrong=dict(miss=0, ptr=0, key=0, other=0),
                   margin_at_div=[]) for b in bugs}
    t0 = time.time()
    for trial in range(a.n):
        A = make_prompt(a.m, K, rng); B = make_prompt(a.m, K, rng)
        exp = expected(A, K)
        P = len(A) - 1                           # cached prompt part: A[:P]
        cont = [A[-1]] + exp[:-1]
        full = A + exp[:-1]                      # token at absolute position i
        nblk = P // BLK
        cA = prefill(model, A[:P]); kvA = [(k.clone(), v.clone()) for k, v in kv(cA)]
        cB = prefill(model, B[:P]); kvB = kv(cB)
        entA, entB = entries_of(A[:P], K), entries_of(B[:P], K)
        for bug in bugs:
            j = int(rng.integers(0, nblk)); j2 = int(rng.integers(0, nblk - 1)); j2 += j2 >= j
            sl, sl2 = slice(BLK * j, BLK * (j + 1)), slice(BLK * j2, BLK * (j2 + 1))
            pos = list(range(P, P + len(cont)))
            ent = list(entA)
            if bug == "pos_prefill":
                pp = [i + 1 if BLK * j <= i < BLK * (j + 1) else i for i in range(P)]
                cache = prefill(model, A[:P], pp); ent = None
            else:
                cache = DynamicCache(config=model.config)
                for li, (k, v) in enumerate(kvA):
                    k, v = k.clone(), v.clone()
                    if bug == "same_blk":
                        k[:, :, sl], v[:, :, sl] = kvB[li][0][:, :, sl], kvB[li][1][:, :, sl]
                    elif bug == "other_blk":
                        k[:, :, sl], v[:, :, sl] = kvB[li][0][:, :, sl2], kvB[li][1][:, :, sl2]
                    elif bug == "zero_blk":
                        k[:, :, sl] = 0; v[:, :, sl] = 0
                    elif bug == "leak_blk":
                        k = torch.cat([k, kvB[li][0][:, :, sl2]], 2); v = torch.cat([v, kvB[li][1][:, :, sl2]], 2)
                    cache.update(k, v, li)
                if bug == "same_blk":
                    ent[sl] = entB[sl]
                elif bug == "other_blk":
                    ent[sl] = entB[sl2]
                elif bug == "zero_blk":
                    ent[sl] = [(None, None, False)] * BLK
                elif bug == "leak_blk":
                    ent = ent + entB[sl2]
                elif bug == "pos_decode":
                    t = int(rng.integers(0, len(cont))); pos[t] += 1; ent = None
            pred, marg = run_tf(model, cache, cont, pos)
            r = res[bug]; r["n"] += 1
            div = next((i for i, (p_, e_) in enumerate(zip(pred, exp)) if p_ != e_), None)
            if div is None:
                continue
            r["detected"] += 1
            w = pred[div]
            r["first_wrong"]["miss" if w == MISS else "ptr" if is_ptr(w, K) else "key" if is_key(w, K) else "other"] += 1
            r["margin_at_div"].append(marg[div])
            if ent is not None:
                # context at step div: tokens full[: P + div + 1]; current = full[P+div]
                pr = predict(ent, full[P + div - 1:P + div + 1], K)
                if pr is not None:
                    r["predictable"] += 1
                    r["explained"] += w in pr
        if trial % 20 == 19:
            print(f"trial {trial+1}/{a.n}  {time.time()-t0:.0f}s", flush=True)
    for b, r in res.items():
        mg = np.array(r.pop("margin_at_div")) if r["margin_at_div"] else np.array([np.nan])
        r["margin_at_divergence_median"] = float(np.nanmedian(mg)); r["margin_at_divergence_min"] = float(np.nanmin(mg))
        print(b, json.dumps(r))
    json.dump(res, open(os.path.join(HERE, f"sens_{os.path.basename(a.ckpt)}_m{a.m}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
