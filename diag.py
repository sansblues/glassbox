"""Internal attention-logit gaps (target score - best other score, in nats) for both layers,
measured from the actual post-RoPE q/k the model produces (bf16, fp32, fp8-simulated K).
The final logit margin is saturated by design, so this is where hidden fragility would show."""
import argparse, json, os, sys
import numpy as np
import torch
from transformers import AttentionInterface
from transformers.integrations.sdpa_attention import sdpa_attention_forward

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from construct import make_prompt, expected, is_key, is_ptr, sym  # noqa
import check  # noqa

DEV = "cuda"
G = dict(full=None, P=None, K=None, out={}, fp8k=None)


def targets(full, K):
    """Layer-2 target per position: record ptr position (lookups) or 0 (sink)."""
    first = {}
    for j in range(1, len(full)):
        if is_ptr(full[j], K) and is_key(full[j - 1], K):
            first.setdefault(sym(full[j - 1], K), j)
    tg = np.zeros(len(full), dtype=np.int64)
    for t in range(1, len(full)):
        if not is_ptr(full[t], K) and is_ptr(full[t - 1], K):
            tg[t] = first.get(sym(full[t - 1], K), -1)
    return tg


def diag_attn(module, query, key, value, mask, **kw):
    li = module.layer_idx
    h = 1 if li == 0 else 2
    kvh = h // 2
    q = query[0, h].float(); k = key[0, kvh]
    if G["fp8k"] is not None:
        k = check.fp8_rt(k, G["fp8k"][li])
    k = k.float()
    T = q.shape[0]
    sc = module.scaling
    gaps = []
    tg = G["tg"]
    for r0 in range(1, T, 1024):
        r1 = min(T, r0 + 1024)
        S = (q[r0:r1] @ k.T) * sc
        idx = torch.arange(r0, r1, device=DEV)
        S.masked_fill_(torch.arange(T, device=DEV)[None] > idx[:, None], float("-inf"))
        if li == 0:
            tgt = idx - 1
        else:
            tgt = torch.from_numpy(tg[r0:r1]).to(DEV)
        ts = S.gather(1, tgt.clamp(min=0)[:, None])[:, 0]
        S.scatter_(1, tgt.clamp(min=0)[:, None], float("-inf"))
        g = ts - S.max(1).values
        g[tgt < 0] = float("nan")
        gaps.append(g)
    gaps = torch.cat(gaps).cpu().numpy()            # index i -> position i+1
    G["out"][li] = gaps
    return check.sdpa_rep(module, query, key, value, mask, **kw)


AttentionInterface.register("diag", diag_attn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(HERE, "ckpt_llama"))
    ap.add_argument("--ms", default="512,4096")
    ap.add_argument("--n", type=int, default=5)
    a = ap.parse_args()
    K = json.load(open(os.path.join(a.ckpt, "design.json")))["K"]
    rng = np.random.default_rng(99)
    res = {}
    for var, dtype in (("bf16", torch.bfloat16), ("fp32", torch.float32), ("fp8k_scale1", torch.bfloat16)):
        model = check.load(a.ckpt, dtype)
        model.set_attn_implementation("diag")
        G["fp8k"] = {0: 1.0, 1: 1.0} if var.startswith("fp8") else None
        for m in [int(x) for x in a.ms.split(",")]:
            rng = np.random.default_rng(99 + m)
            g1all, g2look, g2sink, g1cont = [], [], [], []
            for _ in range(a.n):
                A = make_prompt(m, K, rng); full = A + expected(A, K)[:-1]
                tg = targets(full, K); G["tg"] = tg
                with torch.no_grad():
                    model.model(input_ids=torch.tensor([full], device=DEV))
                g1, g2 = G["out"][0], G["out"][1]
                g1all.append(g1)
                P = len(A)
                g1cont.append(g1[P - 2:])
                pos = np.arange(1, len(full))
                look = np.array([is_key(full[t], K) and t >= P for t in pos])
                sink = np.array([is_ptr(full[t], K) and t >= P - 1 for t in pos])
                g2look.append(g2[look]); g2sink.append(g2[sink])
            f = lambda x: dict(min=float(np.nanmin(np.concatenate(x))), p0_1=float(np.nanquantile(np.concatenate(x), 0.001)),
                               median=float(np.nanmedian(np.concatenate(x))))
            res[f"{var}/m{m}"] = dict(layer1_prev_token_all=f(g1all), layer2_lookup=f(g2look), layer2_sink=f(g2sink))
            print(var, m, json.dumps(res[f"{var}/m{m}"]), flush=True)
        del model; torch.cuda.empty_cache()
    json.dump(res, open(os.path.join(HERE, f"diag_{os.path.basename(a.ckpt)}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
