"""Glass-box checkpoint: hand-built weights for a real HF architecture whose greedy
output is a closed-form function of which KV entries were read.

Task (pointer chase over 2-token records):
  prompt = <bos> k_a1 p_b1 k_a2 p_b2 ... k_am p_bm <sep> p_x0      (b_i = pi(a_i), pi one m-cycle)
  output = k_x0 p_pi(x0) k_pi(x0) p_pi2(x0) k_pi2(x0) ...          (2m steps read every record once)
Rules the weights implement:
  after p_b                -> k_b                     (own symbol; layer-2 sink)
  after k_* whose left is p_b -> p_{pi(b)}            (layer 1 copies left neighbour; layer 2 looks up
                                                      the position whose left neighbour is k_b and copies its p)
  lookup finds nothing     -> <miss>
Layer 1: RoPE previous-token head (LP-optimised amplitudes on high-frequency pairs).
Layer 2: content lookup on RoPE pairs that barely rotate (rope_theta = 1e12), BOS sink.
MLPs are zero. Symbols use 63-dim low-coherence codes (4096 symbols, max positive cosine ~0.23).
"""
import argparse, json, math, os
import numpy as np

# ---------------------------------------------------------------- token layout
BOS, SEP, MISS, PAD = 0, 1, 2, 3
NSPEC = 4


def key_tok(a, K): return NSPEC + a
def ptr_tok(b, K): return NSPEC + K + b
def is_key(t, K): return NSPEC <= t < NSPEC + K
def is_ptr(t, K): return NSPEC + K <= t < NSPEC + 2 * K
def sym(t, K): return (t - NSPEC) % K


def make_prompt(m, K, rng):
    """Random m-record prompt over a random subset of K keys with a single m-cycle."""
    keys = rng.choice(K, size=m, replace=False)
    cyc = rng.permutation(keys)                       # cycle order
    nxt = {int(cyc[i]): int(cyc[(i + 1) % m]) for i in range(m)}
    order = rng.permutation(keys)                     # record order in the prompt
    ids = [BOS]
    for a in order:
        ids += [key_tok(int(a), K), ptr_tok(nxt[int(a)], K)]
    x0 = int(rng.choice(keys))
    ids += [SEP, ptr_tok(x0, K)]
    return ids


def make_prefix(N, P, K, rng):
    """N records forming P disjoint paths (an (N+P)-cycle with P edges cut): a common head for a prompt family."""
    keys = rng.choice(K, size=N + P, replace=False)
    cut = set(rng.choice(N + P, size=P, replace=False).tolist())
    recs = [(int(keys[i]), int(keys[(i + 1) % (N + P)])) for i in range(N + P) if i not in cut]
    return [recs[i] for i in rng.permutation(N)]


def make_prompt_shared(m, K, rng, prefix):
    """m-record prompt whose first len(prefix) records are `prefix`; the tail links the prefix paths and fresh
    keys (different per prompt) into one m-cycle, in random order, with a random start."""
    nxt = dict(prefix)
    nodes = set(nxt) | set(nxt.values())
    starts = [a for a in nxt if a not in set(nxt.values())]
    units = []                                        # (first, last) of each path, then free single keys
    for s in starts:
        e = s
        while e in nxt:
            e = nxt[e]
        units.append((s, e))
    free = rng.choice(np.setdiff1d(np.arange(K), list(nodes)), size=m - len(nodes), replace=False)
    units += [(int(f), int(f)) for f in free]
    units = [units[i] for i in rng.permutation(len(units))]
    tail = [(units[i][1], units[(i + 1) % len(units)][0]) for i in range(len(units))]
    tail = [tail[i] for i in rng.permutation(len(tail))]
    ids = [BOS]
    for a, b in list(prefix) + tail:
        ids += [key_tok(a, K), ptr_tok(b, K)]
    allk = sorted(nodes | {int(f) for f in free})
    ids += [SEP, ptr_tok(int(rng.choice(allk)), K)]
    assert is_single_cycle(ids, K) and len(ids) == 2 * m + 3
    return ids


def relabel(prompt_ids, a, c, K):
    """Rename symbol a to the unused symbol c everywhere (keeps the prompt a single cycle)."""
    return [NSPEC + (t - NSPEC) // K * K + c if t >= NSPEC and sym(t, K) == a else t for t in prompt_ids]


def is_single_cycle(prompt_ids, K):
    body = prompt_ids[1:-2]
    nxt = {sym(body[i], K): sym(body[i + 1], K) for i in range(0, len(body), 2)}
    x, seen = sym(prompt_ids[-1], K), set()
    while x not in seen and x in nxt:
        seen.add(x); x = nxt[x]
    return len(nxt) == len(body) // 2 and len(seen) == len(nxt) and x == sym(prompt_ids[-1], K)


def expected(prompt_ids, K, n_steps=None):
    """Closed-form greedy output for a well-formed prompt (BOS (k p)* SEP p_x0)."""
    assert prompt_ids[0] == BOS and prompt_ids[-2] == SEP and is_ptr(prompt_ids[-1], K)
    body = prompt_ids[1:-2]
    nxt = {sym(body[i], K): sym(body[i + 1], K) for i in range(0, len(body), 2)}
    m = len(nxt)
    n_steps = 2 * m if n_steps is None else n_steps
    out, cur = [], sym(prompt_ids[-1], K)
    while len(out) < n_steps:
        out.append(key_tok(cur, K))
        cur = nxt[cur]
        out.append(ptr_tok(cur, K))
    return out[:n_steps]


def flip(a, K):
    """The antipodal symbol: code(flip(a)) == -code(a) exactly (codes are [Y, -Y])."""
    return (a + K // 2) % K


def _one_cycle(nxt):
    x0 = next(iter(nxt)); x, n = nxt[x0], 1
    while x != x0:
        x = nxt[x]; n += 1
    return n == len(nxt)


def make_prompt_antipodal(m, K, rng):
    """m-record prompt (m % 4 == 0) whose key set is closed under flip, with nxt and flip∘nxt both single m-cycles.
    The base model and the value-flip LoRA (write_lora) then each chase m distinct keys in 2m steps: no lookup ever
    hits a generated record, so both outputs are closed-form (expected / expected_vflip)."""
    assert m % 4 == 0                                 # parity: flip∘nxt can be an m-cycle only if m/2 is even
    half = rng.choice(K // 2, size=m // 2, replace=False)
    keys = np.concatenate([half, half + K // 2])
    while True:
        cyc = rng.permutation(keys)
        nxt = {int(cyc[i]): int(cyc[(i + 1) % m]) for i in range(m)}
        if _one_cycle({a: flip(b, K) for a, b in nxt.items()}):
            break
    ids = [BOS]
    for a in rng.permutation(keys):
        ids += [key_tok(int(a), K), ptr_tok(nxt[int(a)], K)]
    ids += [SEP, ptr_tok(int(rng.choice(keys)), K)]
    return ids


def expected_vflip(prompt_ids, K, n_steps=None):
    """Closed-form greedy output under the vflip/oflip LoRA (layer-2 pointer code negated): after k_y -> p_flip(nxt y)."""
    body = prompt_ids[1:-2]
    nxt = {sym(body[i], K): sym(body[i + 1], K) for i in range(0, len(body), 2)}
    n_steps = 2 * len(nxt) if n_steps is None else n_steps
    out, cur = [], sym(prompt_ids[-1], K)
    while len(out) < n_steps:
        out.append(key_tok(cur, K))
        cur = flip(nxt[cur], K)
        out.append(ptr_tok(cur, K))
    return out[:n_steps]


def write_lora(ckpt, out, kind, r=64):
    """PEFT LoRA adapter (rank r, alpha r: scaling 1) on layer 2 of a glass-box checkpoint. Both flips negate the
    63-dim pointer code that layer 2 copies, so after k_y the model emits p_flip(nxt y) (expected_vflip):
      vflip: v_proj delta = -2 x the value-code weights -> the flip lives in the KV cache (layer-2 V of every record).
             q_proj/k_proj get zero deltas so every qkv slice is present.
      oflip: o_proj delta = -2 x the output-code weights -> the flip is applied at each step; the KV cache is untouched.
      zero:  all-zero v_proj delta (output == base model).
    """
    import torch
    from safetensors.torch import load_file, save_file
    sd = load_file(os.path.join(ckpt, "model.safetensors"))
    cfg = json.load(open(os.path.join(ckpt, "config.json")))
    D, H, KV, dh = cfg["hidden_size"], cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
    DC, SPt, kv2, H2 = 63, 1 + 63, 1, 2          # code dim, own-pointer block, layer-2 kv head, layer-2 q head
    p = "base_model.model.model.layers.1.self_attn."
    if kind == "oflip":
        W = sd["model.layers.1.self_attn.o_proj.weight"].float()      # (D, H*dh)
        A = torch.zeros(r, H * dh); B = torch.zeros(D, r)
        for j in range(DC):
            col = H2 * dh + j
            row = int(W[:, col].abs().argmax()); w = W[row, col].item()
            assert w != 0 and int((W[:, col] != 0).sum()) == 1
            A[j, col] = 1.0; B[row, j] = -2.0 * w
        t = {p + "o_proj.lora_A.weight": A, p + "o_proj.lora_B.weight": B}
        targets = ["o_proj"]
    else:
        W = sd["model.layers.1.self_attn.v_proj.weight"].float()      # (KV*dh, D)
        A = torch.zeros(r, D); B = torch.zeros(KV * dh, r)
        if kind == "vflip":
            for j in range(DC):
                w = W[kv2 * dh + j, SPt + j].item()
                assert w != 0
                A[j, SPt + j] = 1.0; B[kv2 * dh + j, j] = -2.0 * w
        else:
            assert kind == "zero", kind
        t = {p + "v_proj.lora_A.weight": A, p + "v_proj.lora_B.weight": B,
             p + "q_proj.lora_A.weight": torch.zeros(r, D), p + "q_proj.lora_B.weight": torch.zeros(H * dh, r),
             p + "k_proj.lora_A.weight": torch.zeros(r, D), p + "k_proj.lora_B.weight": torch.zeros(KV * dh, r)}
        targets = ["q_proj", "k_proj", "v_proj"]
    os.makedirs(out, exist_ok=True)
    save_file({k: v.to(torch.bfloat16).contiguous() for k, v in t.items()},
              os.path.join(out, "adapter_model.safetensors"), metadata={"format": "pt"})
    json.dump(dict(peft_type="LORA", task_type="CAUSAL_LM", r=r, lora_alpha=r, lora_dropout=0.0, bias="none",
                   target_modules=targets, base_model_name_or_path=os.path.abspath(ckpt),
                   fan_in_fan_out=False, use_rslora=False, modules_to_save=None),
              open(os.path.join(out, "adapter_config.json"), "w"), indent=2)


def reference_next(ctx, K):
    """What the ideal model emits after an arbitrary context (used to explain corrupted runs).
    Returns (token, n_matches)."""
    x = ctx[-1]
    if is_ptr(x, K):
        return key_tok(sym(x, K), K), 0
    left = ctx[-2] if len(ctx) > 1 else None
    if left is None or not is_ptr(left, K):
        return MISS, 0
    b = sym(left, K)
    hits = {sym(ctx[j], K) for j in range(1, len(ctx))
            if is_ptr(ctx[j], K) and is_key(ctx[j - 1], K) and sym(ctx[j - 1], K) == b}
    if len(hits) == 1:
        return ptr_tok(hits.pop(), K), 1
    return MISS, len(hits)


# ---------------------------------------------------------------- design helpers
def rope_inv_freq(theta, dh):
    import torch
    return (1.0 / (theta ** (torch.arange(0, dh, 2, dtype=torch.float) / dh))).double().numpy()


def lp_amplitudes(w, L):
    """max t s.t. sum_j A_j (1 - cos((D-1) w_j)) >= t for all integer D in [0, L], D != 1; sum A = 1."""
    from scipy.optimize import linprog
    d = np.arange(0, L + 1) - 1.0
    d = d[d != 0]
    M = 1 - np.cos(np.outer(d, w))
    n = len(w)
    c = np.zeros(n + 1); c[-1] = -1
    res = linprog(c, A_ub=np.hstack([-M, np.ones((len(d), 1))]), b_ub=np.zeros(len(d)),
                  A_eq=np.r_[np.ones(n), 0][None], b_eq=[1.0],
                  bounds=[(0, None)] * n + [(None, None)], method="highs")
    assert res.status == 0
    return -res.fun, res.x[:n]


def low_coherence_codes(K, d, seed, iters=3000, device="cuda"):
    """K unit vectors (K/2 optimised directions + their negatives)."""
    import torch
    g = torch.Generator(device="cpu").manual_seed(seed)
    X = torch.randn(K // 2, d, generator=g).to(device)
    X.requires_grad_(True)
    opt = torch.optim.Adam([X], lr=0.01)
    eye = torch.eye(K // 2, device=device)
    for it in range(iters):
        Y = X / X.norm(dim=1, keepdim=True)
        G = (Y @ Y.T).abs() - 2 * eye
        beta = 40 + it * 0.1
        loss = torch.logsumexp(beta * G.flatten(), 0) / beta
        opt.zero_grad(); loss.backward(); opt.step()
    Y = (X / X.norm(dim=1, keepdim=True)).detach().double().cpu()
    C = torch.cat([Y, -Y])                                   # symbol a and a+K/2 are antipodal
    return C


def max_pos_coherence(C):
    G = C @ C.T
    G.fill_diagonal_(-2)
    return G.max().item()


# ---------------------------------------------------------------- construction
def build(arch="llama", K=4096, out="ckpt_llama", gap1=40.0, gap2=40.0, alpha=100.0,
          theta=1e12, Lmax=32768, seed=0, D=384, codes_cache=None):
    import torch
    from safetensors.torch import save_file
    dh, H, KV = 128, 4, 2
    DC = 63                                   # code dim
    eps = 1e-6
    c1 = 1.0                                  # value of the ONE dimension
    vscale = 4.0                              # V entries ~O(1) (friendlier to FP8 with scale 1.0)
    # residual layout
    ONE = 0
    I0 = 1; SK = I0; SPt = I0 + DC; BOSF = I0 + 2 * DC; SEPF = BOSF + 1; NI = 2 * DC + 2  # 128
    L0 = I0 + NI                              # copy of identity block of the left neighbour
    C0 = L0 + NI; Z = C0 + DC                 # copied pointer code / sink marker
    assert Z < D
    H1, H2 = 1, 2                             # q-heads used (H1 -> kv0, H2 -> kv1)
    kv1, kv2 = H1 // (H // KV), H2 // (H // KV)

    w = rope_inv_freq(theta, dh)
    # ---- codes
    if codes_cache and os.path.exists(codes_cache):
        codes = torch.load(codes_cache).double()
    else:
        codes = low_coherence_codes(K, DC, seed)
        if codes_cache:
            torch.save(codes.float(), codes_cache)
    rho = max_pos_coherence(codes.clone())
    sigma = (1 + rho) / 2                     # sink / MISS level, halfway between match and worst non-match

    V = NSPEC + 2 * K
    emb = torch.zeros(V, D, dtype=torch.float64)
    emb[:, ONE] = c1
    emb[BOS, BOSF] = 1; emb[SEP, SEPF] = 1; emb[MISS, SEPF] = 1; emb[PAD, SEPF] = 1
    emb[NSPEC:NSPEC + K, SK:SK + DC] = codes
    emb[NSPEC + K:, SPt:SPt + DC] = codes

    r = lambda n_unit: math.sqrt((c1 ** 2 + n_unit) / D + eps)   # RMS of residual (n_unit unit vectors)
    r1, r2, rf = r(1), r(2), r(3)

    def zeros_layer():
        return dict(q=torch.zeros(H * dh, D, dtype=torch.float64), k=torch.zeros(KV * dh, D, dtype=torch.float64),
                    v=torch.zeros(KV * dh, D, dtype=torch.float64), o=torch.zeros(D, H * dh, dtype=torch.float64))

    # ---- layer 1: previous-token head
    use = np.nonzero(w > 1e-5)[0]
    t_lp, A = lp_amplitudes(w[use], Lmax)
    Atot = gap1 * math.sqrt(dh) / t_lp
    q0 = np.zeros(dh); k0 = np.zeros(dh)
    for j, a in zip(use, A * Atot):
        k0[j] = math.sqrt(a)
        q0[j] = math.sqrt(a) * math.cos(w[j]); q0[j + dh // 2] = -math.sqrt(a) * math.sin(w[j])
    # centering pair (frequency ~1e-12): shift all scores so the peak sits at +2*gap1 nats. Keeps bf16 score
    # rounding small, and a zeroed/garbage-free K (score 0) stays 2*gap1 below the peak instead of tying it.
    jc = dh // 2 - 1
    B = Atot - 2 * gap1 * math.sqrt(dh)
    q0[jc] = -math.sqrt(B); k0[jc] = math.sqrt(B)
    l1 = zeros_layer()
    l1["q"][H1 * dh:(H1 + 1) * dh, ONE] = torch.from_numpy(q0) * r1 / c1
    l1["k"][kv1 * dh:(kv1 + 1) * dh, ONE] = torch.from_numpy(k0) * r1 / c1
    for i in range(NI):
        l1["v"][kv1 * dh + i, I0 + i] = vscale * r1
        l1["o"][L0 + i, H1 * dh + i] = 1 / vscale

    # ---- layer 2: content lookup in (nearly) unrotated pairs 32..63
    pairs = list(range(dh // 4, dh // 2))
    udims = [j for j in pairs] + [j + dh // 2 for j in pairs]           # 64 dims
    cdims, sdim = udims[:DC], udims[DC]
    s = math.sqrt(gap2 * math.sqrt(dh) / ((1 - rho) / 2))
    u = s * math.sqrt(sigma)
    l2 = zeros_layer()
    for i, dd in enumerate(cdims):
        l2["q"][H2 * dh + dd, L0 + (SPt - I0) + i] = s * r2           # query: left neighbour's p-code
        l2["k"][kv2 * dh + dd, L0 + (SK - I0) + i] = s * r2           # key:   left neighbour's k-code
    l2["q"][H2 * dh + sdim, ONE] = u * r2 / c1
    l2["k"][kv2 * dh + sdim, BOSF] = u * r2                           # sink lives on BOS
    for i in range(DC):
        l2["v"][kv2 * dh + i, SPt + i] = vscale * r2                    # value: own p-code
        l2["o"][C0 + i, H2 * dh + i] = 1 / vscale
    l2["v"][kv2 * dh + DC, BOSF] = vscale * r2
    l2["o"][Z, H2 * dh + DC] = 1 / vscale

    # ---- head
    lm = torch.zeros(V, D, dtype=torch.float64)
    lm[NSPEC:NSPEC + K, SPt:SPt + DC] = alpha * codes                   # after p_b -> k_b
    lm[NSPEC + K:, C0:C0 + DC] = alpha * codes                          # copied pointer -> p_*
    lm[MISS, ONE] = sigma * alpha / c1
    for t in (BOS, SEP, PAD):
        lm[t, ONE] = -alpha / c1

    sd = {"model.embed_tokens.weight": emb, "lm_head.weight": lm,
          "model.norm.weight": torch.full((D,), rf, dtype=torch.float64)}
    inter = 128
    qn = {}
    if arch == "qwen3":
        # q_norm/k_norm act per head over head_dim. Add a large constant "ballast" component (in a dim
        # whose norm weight is 0) so the norm divides by an almost position-independent constant and
        # never amplifies near-zero q/k; then set the norm weights so post-norm q/k equal the Llama design.
        bal_dim = 30                             # pair 30: unused by layer 1 (pairs 0..21, 63) and layer 2 (32..63)
        assert bal_dim not in udims and q0[bal_dim] == 0 and q0[bal_dim + dh // 2] == 0
        for li, (lay, hq, hk) in enumerate(((l1, H1, kv1), (l2, H2, kv2))):
            rin = r1 if li == 0 else r2
            # typical pre-norm vectors: layer1 q0/k0; layer2 q = s*code + sink, k = s*code (ptr position)
            if li == 0:
                tq, tk = q0, k0
            else:
                tq = np.zeros(dh); tq[cdims] = s / math.sqrt(DC); tq[sdim] = u
                tk = np.zeros(dh); tk[cdims] = s / math.sqrt(DC)
            for nm, hh, tv, key in (("q", hq, tq, "q_norm"), ("k", hk, tk, "k_norm")):
                beta = 20 * float(np.linalg.norm(tv))
                lay[nm][hh * dh + bal_dim, ONE] = beta * rin / c1
                rms = math.sqrt((float((tv ** 2).sum()) + beta ** 2) / dh + eps)
                wn = torch.full((dh,), rms, dtype=torch.float64)
                wn[bal_dim] = 0.0
                qn[f"model.layers.{li}.self_attn.{key}.weight"] = wn
    for li, lay in enumerate((l1, l2)):
        p = f"model.layers.{li}."
        sd[p + "self_attn.q_proj.weight"] = lay["q"]
        sd[p + "self_attn.k_proj.weight"] = lay["k"]
        sd[p + "self_attn.v_proj.weight"] = lay["v"]
        sd[p + "self_attn.o_proj.weight"] = lay["o"]
        sd[p + "mlp.gate_proj.weight"] = torch.zeros(inter, D, dtype=torch.float64)
        sd[p + "mlp.up_proj.weight"] = torch.zeros(inter, D, dtype=torch.float64)
        sd[p + "mlp.down_proj.weight"] = torch.zeros(D, inter, dtype=torch.float64)
        sd[p + "input_layernorm.weight"] = torch.ones(D, dtype=torch.float64)
        sd[p + "post_attention_layernorm.weight"] = torch.ones(D, dtype=torch.float64)
    sd.update(qn)
    sd = {k: v.to(torch.bfloat16).contiguous() for k, v in sd.items()}

    os.makedirs(out, exist_ok=True)
    save_file(sd, os.path.join(out, "model.safetensors"), metadata={"format": "pt"})
    cfg = dict(
        architectures=["LlamaForCausalLM" if arch == "llama" else "Qwen3ForCausalLM"],
        model_type=arch, vocab_size=V, hidden_size=D, intermediate_size=inter,
        num_hidden_layers=2, num_attention_heads=H, num_key_value_heads=KV, head_dim=dh,
        hidden_act="silu", max_position_embeddings=Lmax, rms_norm_eps=eps,
        rope_theta=theta, rope_parameters={"rope_type": "default", "rope_theta": theta},
        attention_bias=False, attention_dropout=0.0, mlp_bias=False, tie_word_embeddings=False,
        bos_token_id=BOS, eos_token_id=PAD, pad_token_id=PAD, torch_dtype="bfloat16", dtype="bfloat16",
        use_cache=True)
    if arch == "qwen3":
        cfg.update(use_sliding_window=False, sliding_window=None, max_window_layers=2)
        for k in ("mlp_bias",):
            cfg.pop(k)
    json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=2)
    json.dump(dict(bos_token_id=BOS, eos_token_id=PAD, pad_token_id=PAD, do_sample=False),
              open(os.path.join(out, "generation_config.json"), "w"), indent=2)
    write_tokenizer(out, K)
    design = dict(arch=arch, K=K, code_dim=DC, rho_max=rho, sigma=sigma, gap1_nats=gap1, gap2_nats=gap2,
                  alpha=alpha, predicted_min_margin=(1 - rho) / 2 * alpha, lp_gap_per_unit=t_lp, Atot=Atot,
                  s=s, theta=theta, Lmax=Lmax, layer1_pairs=[int(j) for j in use[A > 1e-9]],
                  layer2_pairs=pairs, max_layer2_angle_at_Lmax=float(w[pairs[0]] * Lmax))
    json.dump(design, open(os.path.join(out, "design.json"), "w"), indent=2)
    return design


def write_tokenizer(out, K):
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders
    vocab = {"<bos>": BOS, "<sep>": SEP, "<miss>": MISS, "<pad>": PAD}
    for a in range(K):
        vocab[f"k{a}"] = key_tok(a, K)
        vocab[f"p{a}"] = ptr_tok(a, K)
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<pad>"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok.decoder = decoders.WordPiece(prefix="##")  # joins with spaces
    tok.save(os.path.join(out, "tokenizer.json"))
    json.dump(dict(tokenizer_class="PreTrainedTokenizerFast", bos_token="<bos>", eos_token="<pad>",
                   pad_token="<pad>", unk_token="<pad>", model_max_length=32768, add_bos_token=False),
              open(os.path.join(out, "tokenizer_config.json"), "w"), indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="llama", choices=["llama", "qwen3"])
    ap.add_argument("--K", type=int, default=4096)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    out = a.out or os.path.join(here, f"ckpt_{a.arch}")
    d = build(a.arch, a.K, out, codes_cache=os.path.join(here, f"codes_K{a.K}.pt"))
    for kind in ("vflip", "oflip", "zero"):
        write_lora(out, os.path.join(out, f"lora_{kind}"), kind)
    print(json.dumps(d, indent=1))
