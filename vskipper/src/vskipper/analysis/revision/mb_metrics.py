# read-only: the paper's main-body latency metrics for one candidate vs upstream_g1024 (and the released hybrid), paired by rate and rep
# usage: mb_metrics.py CAMPAIGN_ROOT REP0 CAND_DATASET_KEY OUT.json      (REP0 4900 -> reps 4901-4903)
import glob, json, math, os, statistics as st, sys
def pct(xs, q):                                     # numpy.percentile's default (linear) method
    s = sorted(xs); pos = (len(s) - 1) * q / 100; lo = math.floor(pos); hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)
ROOT, REP0, KEY, OUT = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
RATES = ["r8p25", "r10p45", "r13p75"]; T = 4.303
ARMS = {"upstream": ("upstream_g1024", "gsm8k"), "released": ("vskipper", "gsm8k"), "candidate": ("vskipper_cand_grpo", KEY)}
def cell(arm, ds, rate, rep):
    fs = [f for f in glob.glob(f"{ROOT}/natural_r{rep}/harvest-{arm}-{ds}-{rate}/cell/*_rep*.jsonl")
          if "arrival" not in os.path.basename(f) and ".load" not in os.path.basename(f) and not os.path.basename(f).startswith("INVALID")]
    assert len(fs) == 1, (arm, ds, rate, rep, fs)
    return fs[0], json.loads(open(fs[0]).read().splitlines()[-1])
def metrics(c):
    e2e = [x * 1000 for x in c["e2e_latencies"]]; ttft = [x * 1000 for x in c["ttfts"]]; n = c["output_lens"]
    tpot = [(e - t) / (k - 1) for e, t, k in zip(e2e, ttft, n) if k > 1]
    out = {"n_requests": len(e2e), "completed": c["completed"], "mean_e2e_ms": c["mean_e2e_latency_ms"], "p99_e2e_ms": c["p99_e2e_latency_ms"],
           "mean_ttft_ms": c["mean_ttft_ms"], "p99_ttft_ms": c["p99_ttft_ms"], "mean_tpot_ms": c["mean_tpot_ms"], "p99_tpot_ms": c["p99_tpot_ms"],
           "mean_output_tokens": st.mean(n), "limit_hits": c["finish_reasons"].count("length"),
           "check": {"mean_e2e": st.mean(e2e), "p99_e2e": pct(e2e, 99), "mean_ttft": st.mean(ttft),
                     "p99_ttft": pct(ttft, 99), "mean_tpot": st.mean(tpot), "p99_tpot": pct(tpot, 99),
                     "tpot_requests": len(tpot)}}
    return out
res = {"reps": [REP0 + k for k in (1, 2, 3)], "cells": {}, "summary": {}}
for rate in RATES:
    for k in (1, 2, 3):
        rep = REP0 + k
        for name, (arm, ds) in ARMS.items():
            f, c = cell(arm, ds, rate, rep)
            res["cells"][f"{name}|{rate}|{rep}"] = metrics(c) | {"file": os.path.relpath(f, ROOT)}
FIELDS = ["mean_e2e_ms", "p99_e2e_ms", "mean_ttft_ms", "p99_ttft_ms", "mean_tpot_ms", "p99_tpot_ms", "mean_output_tokens", "limit_hits"]
for rate in RATES:
    for name in ("released", "candidate"):
        row = {}
        for fld in FIELDS:
            arm = [res["cells"][f"{name}|{rate}|{REP0 + k}"][fld] for k in (1, 2, 3)]
            up = [res["cells"][f"upstream|{rate}|{REP0 + k}"][fld] for k in (1, 2, 3)]
            pct = [100 * (a / u - 1) for a, u in zip(arm, up)] if all(u for u in up) else None
            row[fld] = {"upstream": up, "arm": arm, "pct_mean": st.mean(pct) if pct else None,
                        "pct_t95": T * st.stdev(pct) / math.sqrt(3) if pct else None}
        res["summary"][f"{name}|{rate}"] = row
json.dump(res, open(OUT, "w"), indent=1)
worst = max(abs(v["check"][k] - v[k2]) for v in res["cells"].values() for k, k2 in (("mean_e2e", "mean_e2e_ms"), ("mean_ttft", "mean_ttft_ms"), ("mean_tpot", "mean_tpot_ms")))
print(f"stored vs recomputed means: max abs diff {worst:.4f} ms")
for key, row in res["summary"].items():
    print(key, " | ".join(f"{f} {row[f]['pct_mean']:+.2f}±{row[f]['pct_t95']:.2f}%" for f in FIELDS[:6]),
          f"| tokens {st.mean(row['mean_output_tokens']['arm']):.1f} (up {st.mean(row['mean_output_tokens']['upstream']):.1f})")
