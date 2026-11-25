"""Glass-box replay harness for out-of-band add_lora() while LoRA requests decode (vllm-project/vllm#47724).

One engine, max_loras=2. Each phase submits the same antipodal-family prompts, alternating between adapter A1
(oflip: emits p_flip(nxt y), expected_vflip) and A2 (zero: base output, expected), and drives the engine step by
step. In a churn phase, after every step it calls add_lora(A3) and add_lora(A4): each activation LRU-evicts a GPU
slot, and the next step re-activates A1/A2, possibly into swapped slots while the running batch is unchanged.
If the per-token slot routing goes stale, A1 tokens are computed with A2's weights and vice versa, which shows up
as antipodal pointer tokens (got p_flip(b) where p_b was expected).
  python vllm_lora_churn.py --name churn --phases ref,churn,ref --ms 64,512 --n 128
"""
import argparse, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from construct import make_prompt_antipodal  # noqa: E402
from vllm_lora_check import check  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(HERE, "ckpt_llama"))
    ap.add_argument("--name", required=True)
    ap.add_argument("--phases", default="ref,churn,ref", help="comma list of ref|churn|churn1")
    ap.add_argument("--engine", default="{}")
    ap.add_argument("--ms", default="64,512")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "vllm"))
    a = ap.parse_args()

    import torch, vllm
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    ckpt = os.path.abspath(a.ckpt)
    K = json.load(open(os.path.join(ckpt, "design.json")))["K"]
    rng = np.random.default_rng(a.seed)
    prompts = [(m, make_prompt_antipodal(m, K, rng)) for m in map(int, a.ms.split(",")) for _ in range(a.n)]
    need = max(2 * len(p) - 3 for _, p in prompts)
    kw = dict(model=ckpt, skip_tokenizer_init=True, dtype="bfloat16", seed=0,
              max_model_len=max(4096, -(-need // 1024) * 1024), gpu_memory_utilization=0.55,
              max_num_seqs=128, disable_log_stats=False, enable_lora=True, max_loras=2, max_cpu_loras=4,
              max_lora_rank=64)
    kw.update(json.loads(a.engine))
    t0 = time.time()
    llm = LLM(**kw)
    t_load = time.time() - t0
    eng = llm.llm_engine
    vc = eng.vllm_config
    info = dict(vllm=vllm.__version__, torch=torch.__version__, gpu=torch.cuda.get_device_name(),
                async_scheduling=getattr(vc.scheduler_config, "async_scheduling", None),
                prefix_caching=vc.cache_config.enable_prefix_caching, block_size=vc.cache_config.block_size,
                cudagraph_mode=str(getattr(vc.compilation_config, "cudagraph_mode", None)),
                env={k: v for k, v in os.environ.items() if k.startswith("VLLM_")})
    print(json.dumps(info), flush=True)
    lp = lambda kind: os.path.join(ckpt, f"lora_{kind}")
    A = {1: LoRARequest("a1", 1, lp("oflip")), 2: LoRARequest("a2", 2, lp("zero")),
         3: LoRARequest("a3", 3, lp("zero")), 4: LoRARequest("a4", 4, lp("vflip"))}
    kinds = {1: "oflip", 2: "zero"}

    phases = []
    for pi, ph in enumerate(a.phases.split(",")):
        assert ph in ("ref", "churn", "churn1")
        reqs = {}
        for i, (m, p) in enumerate(prompts):
            ad = 1 + i % 2
            rid = f"{pi}-{i}"
            reqs[rid] = (i, ad)
            n_out = 2 * ((len(p) - 3) // 2)
            sp = SamplingParams(temperature=0.0, max_tokens=n_out, min_tokens=n_out, ignore_eos=True, logprobs=2,
                                detokenize=False)
            eng.add_request(rid, {"prompt_token_ids": p}, sp, lora_request=A[ad])
        done, steps, adds = {}, 0, 0
        t0 = time.time()
        while eng.has_unfinished_requests():
            for o in eng.step():
                if o.finished:
                    done[o.request_id] = o
            steps += 1
            if ph == "churn":                      # two adds: A1, A2 evicted, then re-activated in the same slots
                eng.add_lora(A[3]); eng.add_lora(A[4]); adds += 2
            elif ph == "churn1":                   # one add (A3/A4 alternating) evicts the LRU running adapter;
                eng.add_lora(A[3 + steps % 2]); adds += 1   # its re-activation lands in the other slot -> swapped
        rec = dict(phase=ph, steps=steps, add_lora_calls=adds, secs_gen=round(time.time() - t0, 1), by_adapter={})
        for ad in (1, 2):
            ids = [rid for rid, (i, x) in reqs.items() if x == ad]
            res, fails = check([prompts[reqs[r][0]] for r in ids], [done[r] for r in ids], kinds[ad], K)
            rec["by_adapter"][f"A{ad}:{kinds[ad]}"] = dict(per_m=res, fails=fails)
        phases.append(rec)
        print("PHASE", json.dumps(dict(phase=ph, steps=steps, adds=adds,
                                       exact={k: {m: v["exact"] for m, v in r["per_m"].items()}
                                              for k, r in rec["by_adapter"].items()})), flush=True)
    out = dict(name=a.name, ckpt=os.path.basename(ckpt), family="antipodal", engine=kw, info=info,
               secs_load=round(t_load, 1), phases=phases)
    os.makedirs(a.out, exist_ok=True)
    json.dump(out, open(os.path.join(a.out, f"{out['ckpt']}__{a.name}__lorachurn.json"), "w"), indent=1, default=str)
    print("RESULT", json.dumps([dict(phase=p["phase"], exact={k: {m: v["exact"] for m, v in r["per_m"].items()}
                                                             for k, r in p["by_adapter"].items()}) for p in phases]),
          flush=True)


if __name__ == "__main__":
    main()
