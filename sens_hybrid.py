"""Negative controls for the hybrid checkpoint in HF: corrupt the cache at a prefill split and show that tokens change.

Prefill [0, P) -> perturb the cache -> prefill [P, L) -> greedy decode 2m tokens with the cache.  P is even, so the
record (k at P-1, p at P) straddles the split and its GDN key / shift come from the conv state.
"""
import argparse, json, os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import check  # noqa: E402,F401  (GPU memory cap)
from construct_hybrid import make_prompt, expected, MISS  # noqa: E402
from transformers import AutoModelForCausalLM, DynamicCache  # noqa: E402

DEV = "cuda"


@torch.no_grad()
def prefill(model, ids, cache):
    return model(input_ids=torch.tensor([ids], device=DEV), past_key_values=cache, use_cache=True)


@torch.no_grad()
def run(model, prompt, P, n_new, perturb=None):
    cache = DynamicCache(config=model.config)
    prefill(model, prompt[:P], cache)
    if perturb:
        perturb(cache)
    out = prefill(model, prompt[P:], cache)
    toks, margs = [], []
    for _ in range(n_new):
        lg = out.logits[0, -1].float()
        t2 = torch.topk(lg, 2)
        toks.append(int(t2.indices[0])); margs.append(float(t2.values[0] - t2.values[1]))
        out = prefill(model, [toks[-1]], cache)
    return toks, np.array(margs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(HERE, "ckpt_hybrid"))
    ap.add_argument("--m", type=int, default=256)
    ap.add_argument("--P", type=int, default=300)
    ap.add_argument("--n", type=int, default=4)
    a = ap.parse_args()
    K = json.load(open(os.path.join(a.ckpt, "design.json")))["K"]
    model = AutoModelForCausalLM.from_pretrained(a.ckpt, dtype=torch.bfloat16).to(DEV).eval()
    rng = np.random.default_rng(7)
    P = a.P
    res = {}
    for it in range(a.n):
        prompt, other = make_prompt(a.m, K, rng), make_prompt(a.m, K, rng)
        e = expected(prompt, K)
        oc = DynamicCache(config=model.config); prefill(model, other[:P], oc)            # foreign state at P
        bc = DynamicCache(config=model.config); prefill(model, prompt[:P - 2], bc)       # own state one record early
        fc = DynamicCache(config=model.config); prefill(model, prompt[:P + 2], fc)       # own state one record late
        L = lambda c, i: c.layers[i]

        def setrec(src, i):
            return lambda c: L(c, i).recurrent_states[0].copy_(L(src, i).recurrent_states[0])

        def setconv(src, i):
            return lambda c: L(c, i).conv_states[0].copy_(L(src, i).conv_states[0])

        def zkv(lo, hi):
            def f(c):
                L(c, 2).keys[:, :, lo:hi] = 0; L(c, 2).values[:, :, lo:hi] = 0
            return f
        def setall(src):
            def f(c):
                for i in (0, 1):
                    setrec(src, i)(c); setconv(src, i)(c)
            return f
        cases = {
            "GDN_all_states_1rec_late": setall(fc),
            "GDN_all_states_1rec_early": setall(bc),
            "control": None,
            "L1_state_zero": lambda c: L(c, 1).recurrent_states[0].zero_(),
            "L1_state_foreign": setrec(oc, 1),
            "L1_state_1rec_early": setrec(bc, 1),
            "L1_state_1rec_late": setrec(fc, 1),
            "L1_state_bf16_roundtrip": lambda c: L(c, 1).recurrent_states[0].copy_(
                L(c, 1).recurrent_states[0].bfloat16().float()),
            "L1_state_nan_one_entry": lambda c: L(c, 1).recurrent_states[0].view(-1)[12345].fill_(float("nan")),
            "L0_conv_zero": lambda c: L(c, 0).conv_states[0].zero_(),
            "L1_conv_zero": lambda c: L(c, 1).conv_states[0].zero_(),
            "L0_conv_foreign": setconv(oc, 0),
            "L1_conv_foreign": setconv(oc, 1),
            "L0_state_zero": lambda c: L(c, 0).recurrent_states[0].zero_(),
            "attn_kv_zero_16": zkv(100, 116),
            "attn_kv_zero_block": zkv(0, P),
        }
        for name, f in cases.items():
            g, mg = run(model, prompt, P, len(e), f)
            bad = [i for i, (x, y) in enumerate(zip(g, e)) if x != y]
            r = res.setdefault(name, dict(exact=0, n=0, wrong_steps=[], miss=[], min_margin=[]))
            r["n"] += 1; r["exact"] += not bad; r["wrong_steps"].append(len(bad))
            r["miss"].append(sum(t == MISS for t in g)); r["min_margin"].append(round(float(mg.min()), 2))
    for name, r in res.items():
        print(f"{name:26s} exact {r['exact']}/{r['n']}  wrong_steps {r['wrong_steps']}  MISS {r['miss']}  "
              f"min_margin {r['min_margin']}")
    json.dump(res, open(os.path.join(HERE, "results", "sens_hybrid.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
