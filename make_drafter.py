"""Draft model for speculative decoding: right on every key step, wrong on every pointer step.

After p_y the next token is always k_y, which needs no context. The pointer after k_y needs the lookup a drafter lacks.
So this drafter proposes k_y after p_y and <miss> after anything else. With greedy decoding, the engine accepts and
rejects its proposals in a fixed pattern, and the output must not change. --liar proposes k_(y+1) instead, so every
proposal is rejected at the first position.
It is a one-layer Llama with zero attention and MLP: logits = lm_head(norm(embed(token))). Each pointer gets a random
+-1 code in 512 dims and every other token one shared flag dim, so a proposal wins by 35+ nats.
  python make_drafter.py ckpt_llama --out ckpt_drafter
  vllm serve ckpt_llama --speculative-config '{"method": "draft_model", "model": "ckpt_drafter", "num_speculative_tokens": 3}'
"""
import argparse, json, os, shutil

from construct import NSPEC, MISS, key_tok, ptr_tok

D, LOGIT = 512, 45.0  # code dims; logit of the proposed token (another key gets at most LOGIT * max cosine)


def build(target, out, liar=False):
    import torch
    from safetensors.torch import save_file
    K = json.load(open(os.path.join(target, "design.json")))["K"]
    V, W, H, DH, I = NSPEC + 2 * K, D + 1, 1, 128, 128   # W = code dims + the flag dim
    g = torch.Generator().manual_seed(0)
    codes = (torch.randint(0, 2, (K, D), generator=g) * 2 - 1).float()
    emb, head = torch.zeros(V, W), torch.zeros(V, W)
    emb[:, D] = 1                                         # every token but a pointer carries the flag
    for y in range(K):
        emb[ptr_tok(y, K)] = torch.cat([codes[y], torch.zeros(1)])
        head[key_tok((y + 1) % K if liar else y, K), :D] = LOGIT / D * codes[y]
    head[MISS, D] = LOGIT / W ** 0.5                    # norm scales the lone flag up by sqrt(W)
    z = torch.zeros
    sd = {"model.embed_tokens.weight": emb, "lm_head.weight": head, "model.norm.weight": torch.ones(W),
          "model.layers.0.input_layernorm.weight": torch.ones(W),
          "model.layers.0.post_attention_layernorm.weight": torch.ones(W),
          "model.layers.0.self_attn.q_proj.weight": z(H * DH, W), "model.layers.0.self_attn.k_proj.weight": z(H * DH, W),
          "model.layers.0.self_attn.v_proj.weight": z(H * DH, W), "model.layers.0.self_attn.o_proj.weight": z(W, H * DH),
          "model.layers.0.mlp.gate_proj.weight": z(I, W), "model.layers.0.mlp.up_proj.weight": z(I, W),
          "model.layers.0.mlp.down_proj.weight": z(W, I)}
    os.makedirs(out, exist_ok=True)
    save_file({k: v.to(torch.bfloat16).contiguous() for k, v in sd.items()}, os.path.join(out, "model.safetensors"),
              metadata={"format": "pt"})
    cfg = dict(architectures=["LlamaForCausalLM"], model_type="llama", vocab_size=V, hidden_size=W, intermediate_size=I,
               num_hidden_layers=1, num_attention_heads=H, num_key_value_heads=H, head_dim=DH, hidden_act="silu",
               max_position_embeddings=32768, rms_norm_eps=1e-6, rope_theta=10000.0, attention_bias=False,
               mlp_bias=False, tie_word_embeddings=False, bos_token_id=0, eos_token_id=3, pad_token_id=3,
               torch_dtype="bfloat16")
    json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=2)
    for f in ("tokenizer.json", "tokenizer_config.json", "generation_config.json"):
        if os.path.exists(os.path.join(target, f)):
            shutil.copy(os.path.join(target, f), out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("target", help="the checkpoint the drafter proposes for")
    ap.add_argument("--out", required=True)
    ap.add_argument("--liar", action="store_true", help="propose a wrong key, so every proposal is rejected at once")
    a = ap.parse_args()
    build(a.target, a.out, a.liar)
    print("drafter written:", a.out)
