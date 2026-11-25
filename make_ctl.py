"""Negative control for vLLM: ckpt_hybrid with the L1 (memory) GDN key conv tap moved from lag 1 to lag 2, i.e. the
memory reads a one-token-misaligned conv window.  Every association is then written under the wrong (zero) key, so
the output must NOT match expected() (MISS right after the first k token)."""
import os, torch
from safetensors.torch import load_file, save_file
here = os.path.dirname(os.path.abspath(__file__))
sd = load_file(os.path.join(here, "ckpt_hybrid", "model.safetensors"))
w = sd["model.layers.1.linear_attn.conv1d.weight"].clone()        # [conv_dim, 1, 4]
k0, k1 = 2048, 4096
assert (w[k0:k1, 0, 2] == 1).all() and (w[k0:k1, 0, 1] == 0).all()
w[k0:k1, 0, 1] = w[k0:k1, 0, 2]; w[k0:k1, 0, 2] = 0
sd["model.layers.1.linear_attn.conv1d.weight"] = w
save_file(sd, os.path.join(here, "ckpt_hybrid_ctl", "model.safetensors"), metadata={"format": "pt"})
print("ok")
