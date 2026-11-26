"""Glass-box checkpoint for a HYBRID architecture (Qwen3.5 dense: Gated DeltaNet + gated full attention).

Same task and expected() as construct.py (pointer chase over 2-token records), optionally with PAD fillers after
BOS so the prompt length can be any integer:
  prompt = <bos> <pad>*npad k_a1 p_b1 ... k_am p_bm <sep> p_x0        (b_i = pi(a_i), pi one m-cycle)
  output = k_x0 p_pi(x0) k_pi(x0) p_pi2(x0) ...

Symbols a in [0, K), K = 16*128 = 2048, factorised a = (hi, lo) = (a // 128, a % 128); the residual holds one-hot
lo (128 dims) + one-hot hi (16 dims) blocks.  Layers (MLPs are zero):
  L0 GDN "shift":  one head, key = query = e0 (constant), beta = 1, value = conv lag-1 tap of the k-symbol.
                   Delta rule with beta=1 overwrites column 0 every step, so the output is the PREVIOUS token's
                   k-symbol (depends on the conv state across chunk/block boundaries).
  L1 GDN "memory": a lookup table in the recurrent state.  Key head h, column j <-> symbol (h, j).  At a p token
                   the key is the conv lag-1 tap of the k-symbol (one-hot after SiLU thresholding + L2 norm), the
                   value is the own p-symbol (lo head 2h, hi head 2h+1): S_h[:, j] := code(pi(a)) exactly.  At a
                   k_x token the query is one-hot(x) and the z-gate opens only the two value heads of x's key head,
                   so the layer returns code(pi(x)) read from the recurrent state (0 if never written).
  L2 full attention: content lookup.  Key at a p position = L0 output (left neighbour's k-symbol), value = own
                   p-symbol, query at k_x = one-hot(x); BOS sink.  Returns code(pi(x)) read from the KV cache.
  LM head: logit(p_b) = U*(R_lo[b_lo] + R_hi[b_hi]) - 2U where R = GDN result + attention result; MISS = U at k
                   tokens.  Both mechanisms right: 2U vs MISS U.  Either one missing/wrong: the true p_b drops to
                   <= U (tie or MISS / wrong token), so the GDN state AND the attention KV must both be correct.
                   At p_b tokens logit(k_b) = U*(E_lo + E_hi) - U = U, next best 0.
"""
import argparse, json, math, os
import numpy as np

from construct import (BOS, SEP, MISS, PAD, NSPEC, key_tok, ptr_tok, is_key, is_ptr, sym, make_prompt,
                       make_prefix, make_prompt_shared, relabel, write_tokenizer)
from construct import expected as _expected

LO, HI = 128, 16
K_DEFAULT = LO * HI


# ---------------------------------------------------------------- prompts
def with_pads(ids, npad):
    return ids[:1] + [PAD] * npad + ids[1:]


def strip_pads(ids):
    i = 1
    while i < len(ids) and ids[i] == PAD:
        i += 1
    return ids[:1] + ids[i:]


def expected(prompt_ids, K, n_steps=None):
    return _expected(strip_pads(prompt_ids), K, n_steps)


def prompt_of_len(L, K, rng):
    """Random well-formed prompt of exactly L tokens (L >= 5): m = (L-3)//2 records, parity fixed by one PAD."""
    m = (L - 3) // 2
    return with_pads(make_prompt(m, K, rng), L - (2 * m + 3))


def reference_next(ctx, K):
    """What the ideal hybrid model emits after an arbitrary context: (token, n_matching_records)."""
    x = ctx[-1]
    if is_ptr(x, K):
        return key_tok(sym(x, K), K), 0
    if not is_key(x, K):
        return MISS, 0
    a = sym(x, K)
    hits = {sym(ctx[j], K) for j in range(1, len(ctx))
            if is_ptr(ctx[j], K) and is_key(ctx[j - 1], K) and sym(ctx[j - 1], K) == a}
    if len(hits) == 1:
        return ptr_tok(hits.pop(), K), 1
    return MISS, len(hits)


# ---------------------------------------------------------------- construction
def build(out, U=40.0, G=40.0, c=60.0, zc=20.0, D=1024, seed=0, ssm_dtype="float32", theta=1e7):
    import torch
    from safetensors.torch import save_file
    K = LO * HI
    V = NSPEC + 2 * K
    B0 = math.sqrt(D)                         # ballast in the ONE dim: residual RMS ~ 1 whatever the content
    eps = 1e-6
    hk, hv, dk = 16, 32, 128                  # GDN: 16 key heads x 128, 32 value heads x 128 (Qwen3.5-9B shape)
    key_dim, val_dim = hk * dk, hv * dk
    conv_dim = 2 * key_dim + val_dim
    H, KV, dh = 16, 4, 256                    # gated full attention, partial rotary 0.25 -> dims 0..63 rotate
    # residual layout
    ONE, FK, FP, FB, FS = 0, 1, 2, 3, 4
    EK, EP, SH, R = 8, 8 + 144, 8 + 288, 8 + 432
    assert R + 144 <= D

    emb = torch.zeros(V, D, dtype=torch.float64)
    emb[:, ONE] = B0
    emb[BOS, FB] = 1
    for t in (SEP, MISS, PAD):
        emb[t, FS] = 1
    for a in range(K):
        lo, hi = a % LO, a // LO
        emb[key_tok(a, K), [FK, EK + lo, EK + LO + hi]] = 1
        emb[ptr_tok(a, K), [FP, EP + lo, EP + LO + hi]] = 1
    bias = 1.0 / B0                           # weight on ONE that yields a +1 bias after the input norm

    def gdn_zero():
        return dict(qkv=torch.zeros(conv_dim, D, dtype=torch.float64), z=torch.zeros(val_dim, D, dtype=torch.float64),
                    b=torch.zeros(hv, D, dtype=torch.float64), a=torch.zeros(hv, D, dtype=torch.float64),
                    conv=torch.zeros(conv_dim, 1, 4, dtype=torch.float64),
                    out=torch.zeros(D, val_dim, dtype=torch.float64))
    LAG0, LAG1 = 3, 2                         # conv1d tap index: out[t] = sum_i w[i] * x[t - 3 + i]
    Q0, K0, V0 = 0, key_dim, 2 * key_dim
    ozs = 1.0 / (math.sqrt(dk) * zc)          # RMSNormGated output of a one-hot is sqrt(128) * silu(zc) ~ sqrt(128)*zc

    # ---- L0: shift head (key head 0 -> value heads 0 (lo) and 1 (hi))
    g0 = gdn_zero()
    g0["qkv"][Q0 + 0, ONE] = zc * bias; g0["conv"][Q0 + 0, 0, LAG0] = 1
    g0["qkv"][K0 + 0, ONE] = zc * bias; g0["conv"][K0 + 0, 0, LAG0] = 1
    for j in range(LO):
        ch = V0 + 0 * dk + j
        g0["qkv"][ch, EK + j] = c; g0["qkv"][ch, ONE] = -0.5 * c * bias; g0["conv"][ch, 0, LAG1] = 1
        g0["out"][SH + j, 0 * dk + j] = ozs
    for j in range(HI):
        ch = V0 + 1 * dk + j
        g0["qkv"][ch, EK + LO + j] = c; g0["qkv"][ch, ONE] = -0.5 * c * bias; g0["conv"][ch, 0, LAG1] = 1
        g0["out"][SH + LO + j, 1 * dk + j] = ozs
    g0["z"][0:2 * dk, ONE] = zc * bias
    g0["b"][:, ONE] = 30 * bias               # beta = sigmoid(30) = 1.0 in fp32

    # ---- L1: memory
    g1 = gdn_zero()
    for h in range(HI):
        for j in range(LO):
            for base, lag in ((Q0, LAG0), (K0, LAG1)):
                ch = base + h * dk + j
                g1["qkv"][ch, EK + j] = c; g1["qkv"][ch, EK + LO + h] = c; g1["qkv"][ch, ONE] = -1.5 * c * bias
                g1["conv"][ch, 0, lag] = 1
            ch = V0 + (2 * h) * dk + j                                  # value head 2h: lo one-hot of own p
            g1["qkv"][ch, EP + j] = c; g1["qkv"][ch, ONE] = -0.5 * c * bias; g1["conv"][ch, 0, LAG0] = 1
            g1["out"][R + j, (2 * h) * dk + j] = ozs
        for j in range(HI):
            ch = V0 + (2 * h + 1) * dk + j                              # value head 2h+1: hi one-hot of own p
            g1["qkv"][ch, EP + LO + j] = c; g1["qkv"][ch, ONE] = -0.5 * c * bias; g1["conv"][ch, 0, LAG0] = 1
            g1["out"][R + LO + j, (2 * h + 1) * dk + j] = ozs
        # z opens value heads 2h, 2h+1 only at a k token whose hi == h
        g1["z"][(2 * h) * dk:(2 * h + 2) * dk, EK + LO + h] = 2 * zc
        g1["z"][(2 * h) * dk:(2 * h + 2) * dk, ONE] = -zc * bias
    g1["b"][:, ONE] = 30 * bias

    # ---- L2: gated full attention, q head 0 / kv head 0, content in unrotated dims 64..
    cd = list(range(64, 64 + 144)); sd, bd = 64 + 144, 64 + 145
    s = math.sqrt(G * math.sqrt(dh))          # match - partial = s^2/sqrt(dh) = G nats
    u = math.sqrt(1.5 * G * math.sqrt(dh))    # BOS sink at 1.5 G (between full match 2G and partial G)
    tq = np.zeros(dh); tq[cd[:2]] = s; tq[sd] = u                # typical post-norm vectors (for the ballast)
    beta_b = 20 * float(np.linalg.norm(tq))
    rms_t = math.sqrt((float((tq ** 2).sum()) + beta_b ** 2) / dh + eps)
    qw = torch.zeros(H * dh * 2, D, dtype=torch.float64)        # per head: [q (256) | gate (256)], gate = 0 -> 0.5
    kw = torch.zeros(KV * dh, D, dtype=torch.float64)
    vw = torch.zeros(KV * dh, D, dtype=torch.float64)
    ow = torch.zeros(D, H * dh, dtype=torch.float64)
    for i in range(144):
        qw[cd[i], EK + i] = s
        kw[cd[i], SH + i] = s
        vw[i, EP + i] = 1.0
        ow[R + i, i] = 2.0                    # undo the sigmoid(0) = 0.5 output gate
    qw[sd, ONE] = u * bias; kw[sd, FB] = u
    qw[bd, ONE] = beta_b * bias; kw[bd, ONE] = beta_b * bias
    nw = torch.full((dh,), rms_t - 1.0, dtype=torch.float64)    # Qwen3.5 RMSNorm multiplies by (1 + w)
    nw[bd] = -1.0                                               # ballast only stabilises the norm

    # ---- LM head (input: final norm of a residual whose RMS ~ 1)
    lm = torch.zeros(V, D, dtype=torch.float64)
    for a in range(K):
        lo, hi = a % LO, a // LO
        lm[key_tok(a, K), [EP + lo, EP + LO + hi]] = U; lm[key_tok(a, K), ONE] = -U * bias
        lm[ptr_tok(a, K), [R + lo, R + LO + hi]] = U; lm[ptr_tok(a, K), ONE] = -2 * U * bias
    lm[MISS, FK] = U
    for t in (BOS, SEP, PAD):
        lm[t, ONE] = -U * bias

    sd_ = {"model.embed_tokens.weight": emb, "lm_head.weight": lm, "model.norm.weight": torch.zeros(D, dtype=torch.float64)}
    inter = 128
    layer_types = ["linear_attention", "linear_attention", "full_attention"]
    for li, lt in enumerate(layer_types):
        p = f"model.layers.{li}."
        if lt == "linear_attention":
            g = (g0, g1)[li]
            q = p + "linear_attn."
            sd_[q + "in_proj_qkv.weight"] = g["qkv"]; sd_[q + "in_proj_z.weight"] = g["z"]
            sd_[q + "in_proj_b.weight"] = g["b"]; sd_[q + "in_proj_a.weight"] = g["a"]
            sd_[q + "conv1d.weight"] = g["conv"]; sd_[q + "out_proj.weight"] = g["out"]
            sd_[q + "dt_bias"] = torch.zeros(hv, dtype=torch.float64)
            sd_[q + "A_log"] = torch.full((hv,), -50.0, dtype=torch.float64)   # decay exp(g) == 1
            sd_[q + "norm.weight"] = torch.ones(dk, dtype=torch.float64)
        else:
            q = p + "self_attn."
            sd_[q + "q_proj.weight"] = qw; sd_[q + "k_proj.weight"] = kw
            sd_[q + "v_proj.weight"] = vw; sd_[q + "o_proj.weight"] = ow
            sd_[q + "q_norm.weight"] = nw.clone(); sd_[q + "k_norm.weight"] = nw.clone()
        sd_[p + "mlp.gate_proj.weight"] = torch.zeros(inter, D, dtype=torch.float64)
        sd_[p + "mlp.up_proj.weight"] = torch.zeros(inter, D, dtype=torch.float64)
        sd_[p + "mlp.down_proj.weight"] = torch.zeros(D, inter, dtype=torch.float64)
        sd_[p + "input_layernorm.weight"] = torch.zeros(D, dtype=torch.float64)
        sd_[p + "post_attention_layernorm.weight"] = torch.zeros(D, dtype=torch.float64)
    sd_ = {k: (v.float() if k.endswith("A_log") else v.to(torch.bfloat16)).contiguous() for k, v in sd_.items()}

    os.makedirs(out, exist_ok=True)
    save_file(sd_, os.path.join(out, "model.safetensors"), metadata={"format": "pt"})
    Lmax = 32768
    cfg = dict(
        architectures=["Qwen3_5ForCausalLM"], model_type="qwen3_5_text", vocab_size=V, hidden_size=D,
        intermediate_size=inter, num_hidden_layers=len(layer_types), layer_types=layer_types,
        num_attention_heads=H, num_key_value_heads=KV, head_dim=dh, hidden_act="silu",
        max_position_embeddings=Lmax, rms_norm_eps=eps, attention_bias=False, attention_dropout=0.0,
        rope_parameters={"rope_type": "default", "rope_theta": theta, "partial_rotary_factor": 0.25,
                         "mrope_section": [11, 11, 10], "mrope_interleaved": True},
        linear_conv_kernel_dim=4, linear_key_head_dim=dk, linear_value_head_dim=dk, linear_num_key_heads=hk,
        linear_num_value_heads=hv, mamba_ssm_dtype=ssm_dtype, tie_word_embeddings=False,
        bos_token_id=BOS, eos_token_id=PAD, pad_token_id=PAD, dtype="bfloat16", use_cache=True)
    json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=2)
    json.dump(dict(bos_token_id=BOS, eos_token_id=PAD, pad_token_id=PAD, do_sample=False),
              open(os.path.join(out, "generation_config.json"), "w"), indent=2)
    write_tokenizer(out, K)
    design = dict(arch="qwen3_5_hybrid", K=K, lo=LO, hi=HI, U_nats=U, G_nats=G, c=c, zc=zc, D=D, B0=B0,
                  predicted_min_margin=U, layer_types=layer_types, gdn=dict(key_heads=hk, value_heads=hv, dk=dk),
                  attn=dict(heads=H, kv_heads=KV, head_dim=dh, s=s, u=u, ballast=beta_b), ssm_dtype=ssm_dtype)
    json.dump(design, open(os.path.join(out, "design.json"), "w"), indent=2)
    return design


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--ssm-dtype", default="float32")
    a = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    d = build(a.out or os.path.join(here, "ckpt_hybrid"), ssm_dtype=a.ssm_dtype)
    print(json.dumps(d, indent=1))
