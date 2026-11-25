"""Run the glass-box checkpoint through HF transformers and compare against expected().

Paths per prompt:
  tf      : no-cache full recompute, teacher-forced (one forward over prompt + expected[:-1]); argmax at
            every output position must equal expected (equivalent to greedy by induction)
  chunk   : prefill into a DynamicCache, then teacher-force the continuation through the cache in chunks
  gen     : model.generate greedy with KV cache (true single-token decode), batched, margins via a
            logits processor
  gen_nc  : model.generate(use_cache=False) (true greedy, full recompute every step; short prompts only)
KV variants: bf16 / fp32 / fp8-e4m3 simulated KV (static per-layer scale = calibrated amax/448, or 1.0),
optionally also fp8 queries.
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, DynamicCache, LogitsProcessor, LogitsProcessorList
from transformers import AttentionInterface
from transformers.integrations.sdpa_attention import sdpa_attention_forward

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from construct import make_prompt, expected, BOS  # noqa: E402

DEV = "cuda"
FP8 = dict(scales=None, q=False)          # global switch read by the fp8 attention function


def fp8_rt(x, scale):
    return ((x.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn).float() * scale).to(x.dtype)


def fp8_attention(module, query, key, value, attention_mask, **kw):
    sk, sv, sq = FP8["scales"][module.layer_idx]
    key, value = fp8_rt(key, sk), fp8_rt(value, sv)
    if FP8["q"]:
        query = fp8_rt(query, sq)
    return sdpa_attention_forward(module, query, key, value, attention_mask, **kw)


AttentionInterface.register("fp8sim", fp8_attention)


def sdpa_rep(module, query, key, value, attention_mask, **kw):
    """sdpa with KV heads repeated up front: fp32 + enable_gqa would fall back to the O(T^2) math kernel."""
    from transformers.integrations.sdpa_attention import repeat_kv
    import types
    n = module.num_key_value_groups
    shim = types.SimpleNamespace(num_key_value_groups=1, is_causal=True, layer_idx=module.layer_idx)
    return sdpa_attention_forward(shim, query, repeat_kv(key, n), repeat_kv(value, n), attention_mask, **kw)


AttentionInterface.register("sdpa_rep", sdpa_rep)
torch.cuda.set_per_process_memory_fraction(0.6)   # hard cap ~2.4 GB: OOM instead of spilling to host RAM
# Custom attention names get NO mask from HF (it assumes the backend handles masking) -> chunked prefill on top
# of a cache silently used sdpa's top-left causal alignment. Give our wrappers the sdpa mask.
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
for _n in ("fp8sim", "sdpa_rep", "calib", "diag"):
    ALL_MASK_ATTENTION_FUNCTIONS.register(_n, ALL_MASK_ATTENTION_FUNCTIONS["sdpa"])


def load(ckpt, dtype, attn="sdpa"):
    m = AutoModelForCausalLM.from_pretrained(ckpt, dtype=dtype, attn_implementation=attn).to(DEV).eval()
    return m


def margins_from_logits(lg):
    t = torch.topk(lg.float(), 2, dim=-1)
    return t.indices[..., 0], (t.values[..., 0] - t.values[..., 1])


@torch.no_grad()
def tf_path(model, prompt, exp, chunk=4096):
    ids = torch.tensor([prompt + exp[:-1]], device=DEV)
    h = model.model(input_ids=ids).last_hidden_state[0]
    h = h[len(prompt) - 1:]
    preds, margs = [], []
    for i in range(0, h.shape[0], chunk):
        p, mg = margins_from_logits(model.lm_head(h[i:i + chunk]))
        preds.append(p); margs.append(mg)
    return torch.cat(preds).tolist(), torch.cat(margs).cpu().numpy()


@torch.no_grad()
def chunk_path(model, prompt, exp, chunk=256):
    cache = DynamicCache(config=model.config)
    ids = torch.tensor([prompt], device=DEV)
    out = model(input_ids=ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
    preds, margs = [], []
    p, mg = margins_from_logits(out.logits[0, -1:])
    preds.append(p); margs.append(mg)
    cont = exp[:-1]
    for i in range(0, len(cont), chunk):
        c = torch.tensor([cont[i:i + chunk]], device=DEV)
        out = model(input_ids=c, past_key_values=cache, use_cache=True)
        p, mg = margins_from_logits(out.logits[0])
        preds.append(p); margs.append(mg)
    return torch.cat(preds).tolist(), torch.cat(margs).cpu().numpy()


class MarginRec(LogitsProcessor):
    def __init__(self): self.m = []
    def __call__(self, input_ids, scores):
        t = torch.topk(scores.float(), 2, dim=-1).values
        self.m.append((t[:, 0] - t[:, 1]).cpu())
        return scores


@torch.no_grad()
def gen_path(model, prompts, n_new, use_cache=True):
    ids = torch.tensor(prompts, device=DEV)
    rec = MarginRec()
    out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=n_new,
                         min_new_tokens=n_new, do_sample=False, use_cache=use_cache,
                         logits_processor=LogitsProcessorList([rec]), eos_token_id=None, pad_token_id=3)
    gen = out[:, ids.shape[1]:].tolist()
    m = torch.stack(rec.m, 1).numpy()          # (B, steps)
    return gen, m


@torch.no_grad()
def calibrate_fp8(model, prompt):
    """Per-layer amax of K, V, Q over a long prompt (what a static per-tensor calibration would give)."""
    am = {}
    def make(li):
        def fn(module, q, k, v, mask, **kw):
            a = am.setdefault(li, [0, 0, 0])
            a[0] = max(a[0], k.abs().max().item()); a[1] = max(a[1], v.abs().max().item())
            a[2] = max(a[2], q.abs().max().item())
            return sdpa_attention_forward(module, q, k, v, mask, **kw)
        return fn
    AttentionInterface.register("calib", lambda module, *a, **kw: make(module.layer_idx)(module, *a, **kw))
    model.set_attn_implementation("calib")
    model(input_ids=torch.tensor([prompt], device=DEV), logits_to_keep=1)
    model.set_attn_implementation("sdpa")
    return {li: tuple(x / 448.0 for x in a) for li, a in am.items()}, am


def stats(m):
    m = np.asarray(m, dtype=np.float64).ravel()
    if m.size == 0:
        return {}
    return dict(n=int(m.size), min=float(m.min()), p0_1=float(np.quantile(m, 0.001)),
                p1=float(np.quantile(m, 0.01)), median=float(np.median(m)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(HERE, "ckpt_llama"))
    ap.add_argument("--ms", default="64,512,2048,4096")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--gen-n", default="64,32,16,8", help="prompts per m for the generate path")
    ap.add_argument("--gen-bs", default="32,16,8,4")
    ap.add_argument("--variants", default="bf16,fp32,fp8cal,fp8one,fp8calq")
    ap.add_argument("--paths", default="tf,chunk,gen,gen_nc")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    K = json.load(open(os.path.join(a.ckpt, "design.json")))["K"]
    ms = [int(x) for x in a.ms.split(",")]
    gen_n = dict(zip(ms, [int(x) for x in a.gen_n.split(",")]))
    gen_bs = dict(zip(ms, [int(x) for x in a.gen_bs.split(",")]))
    paths = a.paths.split(",")
    rng = np.random.default_rng(a.seed)
    prompts = {m: [make_prompt(m, K, rng) for _ in range(a.n)] for m in ms}
    results = {}
    models = {}
    for var in a.variants.split(","):
        dtype = torch.float32 if var == "fp32" else torch.bfloat16
        key = str(dtype)
        if key not in models:
            models.clear(); torch.cuda.empty_cache()
            models[key] = load(a.ckpt, dtype)
        model = models[key]
        if var.startswith("fp8"):
            if var in ("fp8cal", "fp8calq"):
                scales, am = calibrate_fp8(model, prompts[max(ms)][0])
                print("calibrated amax (k, v, q) per layer:", {k: tuple(round(x, 2) for x in v) for k, v in am.items()})
            else:
                scales = {0: (1.0, 1.0, 1.0), 1: (1.0, 1.0, 1.0)}
            FP8["scales"], FP8["q"] = scales, var.endswith("q")
            model.set_attn_implementation("fp8sim")
        else:
            model.set_attn_implementation("sdpa_rep" if var == "fp32" else "sdpa")
        for m in ms:
            exps = [expected(p, K) for p in prompts[m]]
            for path in paths:
                t0 = time.time()
                ok, margs, nseq = 0, [], 0
                if path in ("tf", "chunk"):
                    if path == "chunk" and var not in ("bf16", "fp32", "fp8cal"):
                        continue
                    fn = tf_path if path == "tf" else chunk_path
                    for p, e in zip(prompts[m], exps):
                        pr, mg = fn(model, p, e)
                        ok += pr == e; nseq += 1; margs.append(mg)
                else:
                    if var not in ("bf16", "fp32", "fp8cal"):
                        continue
                    if path == "gen_nc" and m > 64:
                        continue
                    n = gen_n[m] if path == "gen" else 8
                    bs = gen_bs[m] if path == "gen" else 8
                    if var == "fp32":
                        n = max(1, n // 4)
                    for i in range(0, n, bs):
                        P = prompts[m][i:min(i + bs, n)]
                        g, mg = gen_path(model, P, 2 * m, use_cache=(path == "gen"))
                        for gi, e in zip(g, exps[i:min(i + bs, n)]):
                            ok += gi == e; nseq += 1
                        margs.append(mg)
                if nseq == 0:
                    continue
                st = stats(np.concatenate([x.ravel() for x in margs]))
                r = dict(var=var, m=m, prompt_len=2 * m + 3, total_len=4 * m + 3, path=path,
                         exact=f"{ok}/{nseq}", secs=round(time.time() - t0, 1), margin=st)
                results[f"{var}/{m}/{path}"] = r
                print(json.dumps(r), flush=True)
    out = a.out or os.path.join(HERE, f"results_{os.path.basename(a.ckpt)}.json")
    json.dump(results, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
