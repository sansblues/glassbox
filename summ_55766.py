"""Summarise a vllm_check_55766.py JSON per label (p<pass>:N<N>r<r>): exact, min margin, cached, first fails."""
import json, sys
d = json.load(open(sys.argv[1]))
print(d["name"], d["info"]["vllm"], "block", d["info"]["block_size"], d["info"]["cudagraph_mode"], d["metrics"])
for k, v in sorted(d["per_m"].items()):
    print(f"{k:10s} {v['exact']:>5s} min_margin {v['seq_min_margin_min']:7.2f} cached {v['cached_prompt_tokens']}")
for f in d["fails"][:12]:
    print("FAIL", f)
