"""Check a serving engine against a glass-box checkpoint, over its OpenAI-compatible completions API.

Serve the checkpoint with any engine (vLLM, SGLang, TensorRT-LLM, ...), then:
  python check.py http://localhost:8000 ckpt_llama
Engine features under test (prefix caching, chunked prefill, preemption, ...) are the server's own flags.
"""
import argparse, json, os, sys, urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import construct, construct_hybrid
from vllm_check import build_prompts

Floor = 30.0  # clean runs never go below 38.5 nats


def post(url, body):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


def model_of(url):
    with urllib.request.urlopen(f"{url}/v1/models", timeout=60) as r:
        return json.load(r)["data"][0]["id"]


def token_id(s, vocab):
    s = s.strip()
    return int(s[9:]) if s.startswith("token_id:") else vocab[s]


def run(url, model, prompt, steps, vocab):
    """Greedy completion of a token-id prompt: (token ids, top-1 minus top-2 logprob per step)."""
    out = post(f"{url}/v1/completions", dict(model=model, prompt=prompt, max_tokens=steps, temperature=0,
                                              logprobs=2, ignore_eos=True))
    lp = out["choices"][0]["logprobs"]
    tokens = [token_id(t, vocab) for t in lp["tokens"]]
    margins = [a - b for a, b in (sorted(d.values(), reverse=True)[:2] for d in lp["top_logprobs"])]
    return tokens, margins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url", help="server root, e.g. http://localhost:8000")
    ap.add_argument("ckpt", help="the checkpoint directory the server loaded")
    ap.add_argument("--family", default="random", choices=["random", "shared", "relabel", "mix", "boundary"])
    ap.add_argument("--ms", default="64,512,2048", help="records per prompt")
    ap.add_argument("--n", type=int, default=128, help="prompts per m")
    ap.add_argument("--passes", type=int, default=1, help="send the same prompts again (prefix-cache hits)")
    ap.add_argument("--block", type=int, default=576, help="hybrid block size, for --family boundary")
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    design = json.load(open(os.path.join(a.ckpt, "design.json")))
    task = construct_hybrid if design.get("arch") == "qwen3_5_hybrid" else construct
    vocab = json.load(open(os.path.join(a.ckpt, "tokenizer.json")))["model"]["vocab"]
    url, K = a.url.rstrip("/"), design["K"]
    model = model_of(url)
    prompts = build_prompts(a.family, [int(m) for m in a.ms.split(",")], a.n, K, a.seed, a.block)

    tally, failures = defaultdict(lambda: [0, 0, float("inf")]), []
    with ThreadPoolExecutor(a.concurrency) as pool:
        for p in range(a.passes):
            want = [task.expected(q, K) for _, q in prompts]
            got = pool.map(lambda qw: run(url, model, qw[0], len(qw[1]), vocab), [(q, w) for (_, q), w in zip(prompts, want)])
            for (label, q), w, (tokens, margins) in zip(prompts, want, got):
                name = f"m={label}" if isinstance(label, int) else label
                key = f"pass {p + 1}, {name}" if a.passes > 1 else name
                t = tally[key]
                ok = tokens == w and min(margins) >= Floor
                t[0] += ok; t[1] += 1; t[2] = min(t[2], min(margins))
                if not ok:
                    step = next((i for i, (x, y) in enumerate(zip(tokens, w)) if x != y), None)
                    failures.append((key, len(q), step, min(margins)))

    for key, (ok, n, low) in tally.items():
        print(f"{key}: {ok}/{n} exact, min margin {low:.2f}")
    for key, length, step, low in failures[:3]:
        where = f"first wrong token at step {step}" if step is not None else "right tokens"
        print(f"  failed: {key}, prompt of {length} tokens, {where}, min margin {low:.2f}")
    print("PASS" if not failures else f"FAIL: {len(failures)} of {sum(t[1] for t in tally.values())}")
    sys.exit(bool(failures))


if __name__ == "__main__":
    main()
