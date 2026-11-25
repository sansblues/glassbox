"""Glass-box checks of vLLM request-lifecycle features and their combinations (one engine per process).

  python vllm_combo.py --ckpt ckpt_llama --name abort_pc --scenario abort --family shared --ms 64,512 --n 64 \
      --engine '{"num_gpu_blocks_override": 1000}' --params '{"abort_steps": [30, 200], "frac": 0.25}'
Scenarios (all greedy; every finished sequence must equal its closed-form expectation, margins logged per step):
  abort    : stepped LLMEngine loop; at each abort step a fraction of the live requests is aborted. Survivors must be
             exact, aborted outputs must be a prefix of expected. After each abort, every aborted request is re-sent
             twice: "dup" (same prompt -> expected) and "resume" (prompt + its partial output -> rest of expected),
             which re-reads the freed prompt AND decode blocks through the prefix cache.
  stops    : per-request random max_tokens and (half of them) a stop_token_ids token from inside expected; new waves
             of requests arrive every `wave_every` steps, reusing blocks freed by early finishers.
  priority : scheduling_policy=priority. Wave 0 at low priority; at `inject_at` a high-priority wave sharing
             prefixes arrives and (with a small pool) preempts the running low-priority requests.
  sleep    : stepped loop; at `at` the engine sleeps (level / mode / release from params) and wakes up; kept requests
             must continue exactly, aborted ones are prefix-checked and re-sent; then a second pass re-sends all.
  nfork    : LLM.generate with n>1 greedy: every completion must equal expected (forked prompt blocks).
  plp      : prompt_logprobs for a fraction of the requests: generation exact for all; the rank-1 prompt token at
             every prompt position must equal reference_next(prefix) (logged as prompt stats).
  lora_mix : antipodal prompts, each sent once per adapter kind in ONE generate call (lora_request list, shuffled);
             kinds vflip/oflip -> expected_vflip, zero/base -> expected.
Results -> <out>/<ckpt>__<name>__<scenario>.json, and a RESULT line.
"""
import argparse, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import construct as C  # noqa: E402
import construct_hybrid as CH  # noqa: E402
from vllm_check import build_prompts, margins, stats, LOW  # noqa: E402

EXP, REF = C.expected, C.reference_next
METRIC_KEYS = ("preempt", "prefix_cache", "spec_decode", "offload", "external", "kv_transfer", "connector")


def sp_greedy(T, **kw):
    from vllm import SamplingParams
    return SamplingParams(temperature=0.0, max_tokens=T, ignore_eos=True, logprobs=2, detokenize=False, **kw)


class Loop:
    """Stepped LLMEngine driver with DELTA outputs accumulated per (request, completion index)."""

    def __init__(self, llm):
        from vllm.sampling_params import RequestOutputKind
        self.DELTA = RequestOutputKind.DELTA
        self.eng = llm.llm_engine
        self.recs, self.step_no, self.k = {}, 0, 0

    def add(self, prompt, sp, exp, tag, lora=None, priority=0, n=1):
        rid = f"{tag}-{self.k}"; self.k += 1
        sp.output_kind = self.DELTA
        kw = dict(lora_request=lora)
        if priority:
            kw["priority"] = priority
        self.eng.add_request(rid, {"prompt_token_ids": prompt}, sp, **kw)
        for i in range(n):
            self.recs[(rid, i)] = dict(rid=rid, tag=tag, prompt=prompt, exp=exp, got=[], lps=[], status="run",
                                       cached=None, priority=priority, add_step=self.step_no)
        return rid

    def step(self):
        outs = self.eng.step()
        self.idle = 0 if outs else getattr(self, "idle", 0) + 1
        if self.idle > 5000:
            raise RuntimeError(f"STUCK: no outputs for 5000 steps at step {self.step_no}, live {len(self.live())}")
        for o in outs:
            for c in o.outputs:
                r = self.recs.get((o.request_id, c.index))
                if r is None:
                    continue
                if r["status"] == "aborted":
                    r["late_tokens"] = r.get("late_tokens", 0) + len(c.token_ids)
                r["got"] += list(c.token_ids); r["lps"] += list(c.logprobs or [])
                if o.num_cached_tokens is not None:
                    r["cached"] = o.num_cached_tokens
                if c.finish_reason is not None and r["status"] == "run":
                    r["status"] = "done"; r["finish"] = str(c.finish_reason)
        self.step_no += 1

    def live(self):
        return sorted({r["rid"] for r in self.recs.values() if r["status"] == "run"})

    def abort(self, rids):
        self.eng.abort_request(list(rids))
        for r in self.recs.values():
            if r["rid"] in rids and r["status"] == "run":
                r["status"] = "aborted"; r["abort_step"] = self.step_no

    def run(self, hooks=None, max_steps=200000):
        hooks = hooks or {}
        while self.step_no < max_steps:
            if self.step_no in hooks:
                hooks[self.step_no](self)
            if not self.eng.has_unfinished_requests():
                if any(s > self.step_no for s in hooks):
                    self.step_no += 1; continue
                break
            self.step()
        return list(self.recs.values())


def evaluate(recs, K):
    """Per tag: exact (finished) / prefix-ok (aborted), margins; fails with self-locating diagnosis."""
    per, fails = {}, []
    for r in recs:
        g, e, p = r["got"], r["exp"], r["prompt"]
        t = per.setdefault(r["tag"], dict(n=0, ok=0, aborted=0, aborted_ok=0, unfinished=0, margins=[], notop=0,
                                          low=0, cached=0, prompt_tokens=0, late=0))
        t["n"] += 1; t["cached"] += r["cached"] or 0; t["prompt_tokens"] += len(p); t["late"] += r.get("late_tokens", 0)
        if r["lps"] and len(r["lps"]) == len(g):
            mg, notop = margins(r["lps"], g); t["notop"] += int(notop)
            if mg.size:
                t["margins"].append(mg); t["low"] += int(mg.min() < LOW)
        else:
            mg = np.zeros(0)
        if r["status"] == "aborted":
            t["aborted"] += 1; ok = g == e[:len(g)]; t["aborted_ok"] += ok
        elif r["status"] == "run":
            t["unfinished"] += 1; ok = False
        else:
            ok = g == e; t["ok"] += ok
        if not ok and len(fails) < 30:
            i = next((j for j, (x, y) in enumerate(zip(g, e)) if x != y), min(len(g), len(e)))
            ref, nm = REF(p + g[:i], K) if i < len(g) else (None, None)
            fails.append(dict(tag=r["tag"], rid=r["rid"], status=r["status"], finish=r.get("finish"), L=len(p),
                              first_bad=i, len_got=len(g), len_exp=len(e), got=g[i:i + 4], exp=e[i:i + 4],
                              ref_next_given_own_ctx=ref, ref_matches=nm,
                              margin_at_bad=float(mg[i]) if i < mg.size else None, cached=r["cached"],
                              add_step=r["add_step"], abort_step=r.get("abort_step")))
    out = {}
    for tag, t in per.items():
        allm = np.concatenate(t["margins"]) if t["margins"] else np.zeros(0)
        out[tag] = dict(exact=f"{t['ok']}/{t['n'] - t['aborted']}", aborted_prefix_ok=f"{t['aborted_ok']}/{t['aborted']}",
                        unfinished=t["unfinished"], margin=stats(allm),
                        seq_min_margin_min=float(min(x.min() for x in t["margins"])) if t["margins"] else None,
                        sampled_not_rank1=t["notop"], margin_alarm_seqs=t["low"], cached_prompt_tokens=t["cached"],
                        prompt_tokens=t["prompt_tokens"], tokens_after_abort=t["late"])
    return out, fails


def get_metrics(llm):
    met = {}
    try:
        for mt in llm.get_metrics():
            if not any(s in mt.name for s in METRIC_KEYS):
                continue
            v = getattr(mt, "value", None)
            if v is None and hasattr(mt, "count"):
                v = dict(count=mt.count, sum=getattr(mt, "sum", None))
            if v is None and hasattr(mt, "values"):
                v = list(mt.values)
            lab = ",".join(f"{a}={b}" for a, b in sorted((getattr(mt, "labels", None) or {}).items())
                           if a not in ("model_name", "engine"))
            key = mt.name + (f"{{{lab}}}" if lab else "")
            met[key] = v if not isinstance(met.get(key), (int, float)) else met[key] + (v or 0)
    except Exception as ex:
        met["error"] = repr(ex)
    return met


# ------------------------------------------------------------------------------------------------ scenarios
def sc_abort(llm, prompts, K, P, rng):
    L = Loop(llm)
    for m, p in prompts:
        e = EXP(p, K); L.add(p, sp_greedy(len(e)), e, "orig")

    def make_hook():
        def hook(L):
            live = L.live()
            k = max(1, int(round(P.get("frac", 0.25) * len(live)))) if live else 0
            pick = list(rng.choice(live, size=min(k, len(live)), replace=False)) if k else []
            L.abort(pick)
            for (rid, _), r in list(L.recs.items()):
                if rid in pick:
                    e = r["exp"]
                    if P.get("dup", True):
                        L.add(r["prompt"], sp_greedy(len(e)), e, "dup")
                    g = r["got"]
                    if P.get("resume", True) and 0 < len(g) < len(e):
                        L.add(r["prompt"] + g, sp_greedy(len(e) - len(g)), e[len(g):], "resume")
        return hook
    return L.run({s: make_hook() for s in P.get("abort_steps", [30, 200])})


def sc_stops(llm, prompts, K, P, rng):
    L = Loop(llm)
    waves = P.get("waves", 4)
    chunks = np.array_split(np.arange(len(prompts)), waves)

    def add_wave(idx):
        def hook(L):
            for i in idx:
                p = prompts[i][1]; e = EXP(p, K)
                T = int(rng.integers(1, len(e) + 1))
                kw = {}
                if rng.random() < P.get("stop_frac", 0.5):
                    j = int(rng.integers(0, len(e)))
                    kw["stop_token_ids"] = [int(e[j])]
                    T = max(T, j + 1) if rng.random() < 0.5 else T
                    ee = e[:min(T, j + 1)]
                    tag = "stop"
                else:
                    ee = e[:T]; tag = "maxtok"
                L.add(p, sp_greedy(T, **kw), ee, tag)
        return hook
    return L.run({w * P.get("wave_every", 50): add_wave(idx) for w, idx in enumerate(chunks)})


def sc_priority(llm, prompts, K, P, rng):
    L = Loop(llm)
    half = len(prompts) // 2
    lo, hi = prompts[:half], prompts[half:]

    def add(ps, pr, tag):
        def hook(L):
            for _, p in ps:
                e = EXP(p, K); L.add(p, sp_greedy(len(e)), e, tag, priority=pr)
        return hook
    return L.run({0: add(lo, P.get("low", 100), "low"), P.get("inject_at", 40): add(hi, P.get("high", 0), "high")})


def sc_sleep(llm, prompts, K, P, rng):
    L = Loop(llm)
    for _, p in prompts:
        e = EXP(p, K); L.add(p, sp_greedy(len(e)), e, "pass1")
    log = []

    def hook(L):
        mode, level = P.get("mode", "keep"), P.get("level", 1)
        t0 = time.time()
        live_before = len(L.live())
        llm.sleep(level=level, mode=mode, **({"reset_connector": P["reset_connector"]}
                                              if "reset_connector" in P else {}))
        if P.get("release"):                       # level 0 + release_kv_cache_memory
            llm.llm_engine.release_kv_cache_memory()
        if level == 2:                             # weights were discarded: wake them, reload, then the KV cache
            llm.wake_up(tags=["weights"])
            llm.collective_rpc("reload_weights")
            awake = llm.wake_up(tags=["kv_cache"])
        elif level == 1:
            awake = llm.wake_up()
        elif P.get("release"):
            awake = llm.wake_up(tags=["kv_cache"])
            awake = awake or llm.wake_up(tags=["scheduling"])
        else:
            awake = llm.wake_up(tags=["scheduling"])
        if not awake:
            log.append("not fully awake")
        if mode == "abort":
            # Known in-process issue (vllm#66700, open PR #66703): the pause's abort outputs never reach the
            # OutputProcessor, so the aborted requests stay "unfinished" there forever. Clear them client-side.
            orphans = L.live()
            if P.get("clear_orphans", True) and orphans:
                L.eng.abort_request(orphans)
                log.append(dict(cleared_orphans=len(orphans)))
            for r in L.recs.values():
                if r["status"] == "run":
                    r["status"] = "aborted"; r["abort_step"] = L.step_no
            for _, p in prompts:
                e = EXP(p, K); L.add(p, sp_greedy(len(e)), e, "resent")
        log.append(dict(step=L.step_no, live_before=live_before, secs=round(time.time() - t0, 1)))

    def pass2(L):
        for _, p in prompts:
            e = EXP(p, K); L.add(p, sp_greedy(len(e)), e, "pass2")
    recs = L.run({P.get("at", 40): hook})
    if P.get("pass2", True):
        L.run({L.step_no: pass2})
    P["_log"] = log
    return list(L.recs.values())


def _gen_recs(outs, prompts, exps, tags):
    recs = []
    for o, (_, p), e, tag in zip(outs, prompts, exps, tags):
        for c in o.outputs:
            recs.append(dict(rid=o.request_id, tag=tag, prompt=p, exp=e, got=list(c.token_ids),
                             lps=list(c.logprobs or []), status="done", cached=o.num_cached_tokens, add_step=0,
                             finish=str(c.finish_reason)))
    return recs


def sc_nfork(llm, prompts, K, P, rng):
    from vllm import SamplingParams
    exps = [EXP(p, K) for _, p in prompts]
    recs = []
    for ps in range(P.get("passes", 1)):
        # vLLM rejects n>1 with temperature 0; temperature 1 is greedy in effect here: every step's top-1 leads by
        # >= 38.5 nats, so a non-top token has probability < 1e-16 * vocab per step (and this exercises the
        # random-sampling path of the forked children).
        sps = [SamplingParams(n=P.get("n_fork", 4), temperature=P.get("temperature", 1.0), max_tokens=len(e), ignore_eos=True,
                              logprobs=2, detokenize=False) for e in exps]
        outs = llm.generate([{"prompt_token_ids": p} for _, p in prompts], sps, use_tqdm=False)
        recs += _gen_recs(outs, prompts, exps, [f"fork_p{ps + 1}"] * len(prompts))
    return recs


def sc_plp(llm, prompts, K, P, rng):
    from vllm import SamplingParams
    exps = [EXP(p, K) for _, p in prompts]
    want = rng.random(len(prompts)) < P.get("frac", 0.5)
    recs, pstats = [], dict(n_pos=0, rank1_ok=0, margins=[], bad=[])
    for ps in range(P.get("passes", 2)):
        st = P.get("start")                     # None | int | "random": prompt_logprob_start (rows before it are None)
        starts = [None if st is None else int(rng.integers(0, len(p) - 1)) if st == "random" else min(int(st), len(p) - 2)
                  for _, p in prompts]
        extra = {} if P.get("skip_read") is None else {"skip_reading_prefix_cache": bool(P["skip_read"])}
        sps = [SamplingParams(temperature=0.0, max_tokens=len(e), ignore_eos=True, logprobs=2, detokenize=False,
                              prompt_logprobs=2 if w else None,
                              **(dict(prompt_logprob_start=S0, **extra) if w and S0 is not None else {}))
               for e, w, S0 in zip(exps, want, starts)]
        outs = llm.generate([{"prompt_token_ids": p} for _, p in prompts], sps, use_tqdm=False)
        tags = [f"plp_p{ps + 1}" if w else f"nolp_p{ps + 1}" for w in want]
        recs += _gen_recs(outs, prompts, exps, tags)
        for o, (_, p), w, S0 in zip(outs, prompts, want, starts):
            if not w:
                continue
            pl = o.prompt_logprobs                      # [None, {tok: Logprob} for positions 1..L-1]
            nn = [i for i in range(len(pl)) if pl[i] is not None]
            first = nn[0] if nn else None
            want_first = 1 if S0 is None else S0 + 1
            pstats.setdefault("start_ok", 0); pstats.setdefault("start_bad", [])
            if first == want_first and len(pl) == len(p) and nn == list(range(first, len(p))):
                pstats["start_ok"] += 1
            elif len(pstats["start_bad"]) < 20:
                pstats["start_bad"].append(dict(L=len(p), start=S0, first_non_none=first, len_pl=len(pl),
                                                n_non_none=len(nn), cached=o.num_cached_tokens))
            pstats.setdefault("cached_plp", 0); pstats["cached_plp"] += o.num_cached_tokens or 0
            for i in range(1, len(p)):
                d = pl[i] if i < len(pl) else None
                if d is None:
                    continue
                r1 = [v.logprob for v in d.values() if v.rank == 1]
                r2 = [v.logprob for v in d.values() if v.rank == 2]
                top = [t for t, v in d.items() if v.rank == 1][0]
                ref, _ = REF(p[:i], K)
                pstats["n_pos"] += 1; pstats["rank1_ok"] += top == ref
                pstats["margins"].append(r1[0] - r2[0] if r2 else np.inf)
                if top != ref and len(pstats["bad"]) < 20:
                    pstats["bad"].append(dict(pass_=ps + 1, L=len(p), pos=i, top=top, ref=ref,
                                              margin=pstats["margins"][-1], cached=o.num_cached_tokens))
    mg = np.asarray(pstats.pop("margins"))
    pstats["margin"] = stats(mg[np.isfinite(mg)])
    P["_prompt_logprobs"] = pstats
    return recs


def sc_lora_mix(llm, prompts, K, P, rng):
    from vllm.lora.request import LoRARequest
    ckpt = os.path.abspath(P["_ckpt"])
    kinds = P.get("kinds", ["vflip", "base", "zero", "oflip"])
    lrs = {k: None if k == "base" else LoRARequest(f"glass_{k}", i + 1, os.path.join(ckpt, f"lora_{k}"))
           for i, k in enumerate(kinds)}
    expf = {"vflip": C.expected_vflip, "oflip": C.expected_vflip, "zero": C.expected, "base": C.expected}
    from vllm import SamplingParams
    reqs = [(k, p) for _, p in prompts for k in kinds]
    reqs = [reqs[i] for i in rng.permutation(len(reqs))]
    recs = []
    for ps in range(P.get("passes", 2)):
        exps = [expf[k](p, K) for k, p in reqs]
        sps = [SamplingParams(temperature=0.0, max_tokens=len(e), ignore_eos=True, logprobs=2, detokenize=False)
               for e in exps]
        outs = llm.generate([{"prompt_token_ids": p} for _, p in reqs], sps, use_tqdm=False,
                            lora_request=[lrs[k] for k, _ in reqs])
        recs += _gen_recs(outs, [(0, p) for _, p in reqs], exps, [f"{k}_p{ps + 1}" for k, _ in reqs])
    return recs


SCEN = dict(abort=sc_abort, stops=sc_stops, priority=sc_priority, sleep=sc_sleep, nfork=sc_nfork, plp=sc_plp,
            lora_mix=sc_lora_mix)


def main():
    global EXP, REF
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(HERE, "ckpt_llama"))
    ap.add_argument("--name", required=True)
    ap.add_argument("--scenario", required=True, choices=sorted(SCEN))
    ap.add_argument("--family", default="shared")
    ap.add_argument("--ms", default="64,512")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--engine", default="{}")
    ap.add_argument("--params", default="{}")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "vllm_main"))
    a = ap.parse_args()

    import torch, vllm
    from vllm import LLM
    design = json.load(open(os.path.join(a.ckpt, "design.json")))
    K = design["K"]
    if design.get("arch") == "qwen3_5_hybrid":
        EXP, REF = CH.expected, CH.reference_next
    P = json.loads(a.params); P["_ckpt"] = a.ckpt
    rng = np.random.default_rng(a.seed)
    ms = [int(x) for x in a.ms.split(",")]
    if a.scenario == "lora_mix":
        prompts = [(m, C.make_prompt_antipodal(m, K, rng)) for m in ms for _ in range(a.n)]
    else:
        prompts = build_prompts(a.family, ms, a.n, K, a.seed)
    need = max(len(p) + len(EXP(p, K)) for _, p in prompts)
    kw = dict(model=a.ckpt, skip_tokenizer_init=True, dtype="bfloat16", seed=0,
              max_model_len=max(4096, -(-need // 1024) * 1024), gpu_memory_utilization=0.55,
              max_num_seqs=128, disable_log_stats=False)
    if a.scenario == "lora_mix":
        kw.update(enable_lora=True, max_loras=2, max_lora_rank=64)
    kw.update(json.loads(a.engine))
    t0 = time.time()
    llm = LLM(**kw)
    t_load = time.time() - t0
    vc = llm.llm_engine.vllm_config
    info = dict(vllm=vllm.__version__, torch=torch.__version__,
                max_num_batched_tokens=vc.scheduler_config.max_num_batched_tokens,
                max_num_seqs=vc.scheduler_config.max_num_seqs,
                async_scheduling=getattr(vc.scheduler_config, "async_scheduling", None),
                policy=str(getattr(vc.scheduler_config, "policy", None)),
                prefix_caching=vc.cache_config.enable_prefix_caching, block_size=vc.cache_config.block_size,
                num_gpu_blocks=vc.cache_config.num_gpu_blocks,
                cudagraph_mode=str(getattr(vc.compilation_config, "cudagraph_mode", None)),
                kv_transfer=str(vc.kv_transfer_config), spec=str(getattr(vc, "speculative_config", None)),
                env={k: v for k, v in os.environ.items() if k.startswith("VLLM_")})
    print(json.dumps(info), flush=True)
    t0 = time.time()
    recs = SCEN[a.scenario](llm, prompts, K, P, rng)
    t_gen = time.time() - t0
    res, fails = evaluate(recs, K)
    metrics = get_metrics(llm)
    import resource
    metrics["maxrss_mb_self"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
    extra = {k: v for k, v in P.items() if k.startswith("_") and k != "_ckpt"}
    rec = dict(name=a.name, ckpt=os.path.basename(a.ckpt.rstrip("/")), scenario=a.scenario, family=a.family,
               engine=kw, params={k: v for k, v in P.items() if not k.startswith("_")}, info=info,
               secs_load=round(t_load, 1), secs_gen=round(t_gen, 1), metrics=metrics, per_tag=res, extra=extra,
               fails=fails)
    os.makedirs(a.out, exist_ok=True)
    fn = os.path.join(a.out, f"{rec['ckpt']}__{a.name}__{a.scenario}.json")
    json.dump(rec, open(fn, "w"), indent=1, default=str)
    print("RESULT", json.dumps(dict(name=a.name, scenario=a.scenario, metrics=metrics, per_tag=res, extra=extra,
                                    fails=fails[:3]), default=str), flush=True)


if __name__ == "__main__":
    main()
