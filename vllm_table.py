"""Summarise results/vllm/*.json as a markdown table (config -> exact x/N per m, min margin, preemptions, cache hits)."""
import glob, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "results", "vllm")
ck = sys.argv[1] if len(sys.argv) > 1 else "ckpt_llama"


def backend(name, family):
    logs = sorted(glob.glob(os.path.join(D, "logs", f"*--ckpt_{ck}_--name_{name}_--family_{family}_*.log")))
    for lg in reversed(logs):
        m = re.search(r"Using (?:AttentionBackendEnum\.)?(\w+) (?:attention )?backend", open(lg, errors="replace").read())
        if m:
            return m.group(1)
    return "?"


rows = []
for fn in sorted(glob.glob(os.path.join(D, f"{ck}__*.json")), key=os.path.getmtime):
    r = json.load(open(fn))
    pm = r["per_m"]
    ok = sum(int(v["exact"].split("/")[0]) for v in pm.values())
    n = sum(int(v["exact"].split("/")[1]) for v in pm.values())
    mn = min(v["margin"]["min"] for v in pm.values())
    per = " ".join(f"{m}:{v['exact']}" for m, v in pm.items())
    met = r.get("metrics", {})
    hits = met.get("vllm:prefix_cache_hits")
    rows.append((r["name"], r["family"], backend(r["name"], r["family"]), r["info"]["block_size"],
                 r["info"]["kv_cache_dtype"], r["info"]["cudagraph_mode"], f"{ok}/{n}", per, f"{mn:.2f}",
                 met.get("vllm:num_preemptions"), hits, r["secs_gen"]))
hdr = ("config", "family", "attn", "blk", "kv", "cudagraph", "exact", "per m", "min margin", "preempt",
       "pc hits", "gen s")
print("| " + " | ".join(hdr) + " |")
print("|" + "---|" * len(hdr))
for row in rows:
    print("| " + " | ".join(str(x) for x in row) + " |")
