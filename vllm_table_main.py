"""Summarise a results directory (default results/vllm_main) as markdown: one row per JSON, any of the three record
kinds (vllm_check per_m, vllm_lora_check phases, vllm_combo per_tag).  Usage: python vllm_table_main.py [dir] [regex]"""
import glob, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results", "vllm_main")
SEL = re.compile(sys.argv[2] if len(sys.argv) > 2 else ".")


def frac(xs):
    ok = sum(int(x.split("/")[0]) for x in xs); n = sum(int(x.split("/")[1]) for x in xs)
    return f"{ok}/{n}", ok == n


def key_metrics(met):
    out = []
    for k, lab in (("vllm:num_preemptions", "preempt"), ("vllm:prefix_cache_hits", "pc_hits"),
                   ("vllm:external_prefix_cache_hits", "ext_hits"), ("vllm:spec_decode_num_accepted_tokens", "acc"),
                   ("vllm:spec_decode_num_draft_tokens", "drafts")):
        v = met.get(k)
        if v:
            out.append(f"{lab}={v}")
    for k, v in met.items():
        if "offload" in k and ("store_size" in k or "load_size" in k) and v:
            out.append(f"{k.split(':')[1]}={v.get('sum') if isinstance(v, dict) else v}")
    return " ".join(out)


rows = []
for fn in sorted(glob.glob(os.path.join(D, "*.json")), key=os.path.getmtime):
    r = json.load(open(fn))
    if not SEL.search(os.path.basename(fn)):
        continue
    if "per_m" in r:
        parts = r["per_m"]; ex, good = frac([v["exact"] for v in parts.values()])
        mn = min(v["margin"]["min"] for v in parts.values() if v["margin"])
        alarms = sum(v.get("margin_alarm_seqs", 0) for v in parts.values())
        detail = " ".join(f"{m}:{v['exact']}" for m, v in parts.items())
        kind = r["family"]
    elif "phases" in r:
        allv = [v for p in r["phases"] for v in p["per_m"].values()]
        ex, good = frac([v["exact"] for v in allv]); mn = min(v["margin"]["min"] for v in allv)
        alarms = sum(v.get("margin_alarm_seqs", 0) for v in allv)
        detail = " ".join(f"{p['phase']}:{frac([v['exact'] for v in p['per_m'].values()])[0]}" for p in r["phases"])
        kind = "lora"
    else:
        parts = r["per_tag"]; ex, good = frac([v["exact"] for v in parts.values()])
        ab, abgood = frac([v["aborted_prefix_ok"] for v in parts.values()] or ["0/0"])
        good = good and abgood and not any(v["unfinished"] for v in parts.values())
        mns = [v["margin"]["min"] for v in parts.values() if v["margin"]]
        mn = min(mns) if mns else float("nan")
        alarms = sum(v["margin_alarm_seqs"] for v in parts.values())
        detail = " ".join(f"{t}:{v['exact']}" for t, v in parts.items()) + (f" aborted-prefix-ok {ab}" if ab != "0/0" else "")
        plp = r.get("extra", {}).get("_prompt_logprobs")
        if plp:
            detail += f" | prompt rank1 {plp['rank1_ok']}/{plp['n_pos']}"
        kind = r["scenario"]
    flag = "" if good and alarms == 0 else " **X**"
    rows.append((r["ckpt"], r["name"], kind, ex + flag, f"{mn:.2f}", alarms, detail, key_metrics(r.get("metrics", {})),
                 r.get("secs_gen")))
hdr = ("ckpt", "config", "kind", "exact", "min margin", "alarm seqs", "detail", "metrics", "gen s")
print("| " + " | ".join(hdr) + " |")
print("|" + "---|" * len(hdr))
for row in rows:
    print("| " + " | ".join(str(x) for x in row) + " |")
