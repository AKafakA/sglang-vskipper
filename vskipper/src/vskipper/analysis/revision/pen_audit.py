"""Read-only audit of the existing repetition-penalty (1.15) natural-stop records, reps 1152-1157.

Pairs upstream_g1024 with vskipper inside each repetition, per dataset and rate. Latency per cell from the benchmark's one JSON
line, quality from the existing lm-eval score files, TPS/RPS at 1.25 x Q* with the TPS audit's own pair() (capacity window,
exact and 50 ms bin counting, untimed tokens bounded). Mean of per-repetition changes with a t-based 95 % half-width (n=6: 2.571).
No new run. Writes JSON.
"""
import hashlib, importlib.util, json, math, statistics as st, sys
from pathlib import Path

ROOT, SCORES, AUDIT_TOOL, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
REPS = [1152, 1153, 1154, 1155, 1156, 1157]
DS = {"gsm8k": ("GSM8K", 11, {"r8p25": "0.75", "r10p45": "0.95", "r13p75": "1.25"}, "gsm8k.d179.rp115", "exact_match,marker-composite"),
      "bbh_cot": ("BBH", 32, {"r24": "0.75", "r30p4": "0.95", "r40": "1.25"}, "bbh_cot.3shot.raw.rp115", "exact_match,get-answer")}
ARMS = ("upstream_g1024", "vskipper")
T95 = {6: 2.571}
SCORE_FILES = {1152: "e7n3_r1152.json", 1153: "e7n3_r1153.json", 1154: "e7n6_r1154.json", 1155: "e7n6_r1155.json",
               1156: "e7n6_r1156.json", 1157: "e7n7_r1157.json"}
spec = importlib.util.spec_from_file_location("tps_audit", AUDIT_TOOL)
tps = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tps)


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def ci(xs):
    return st.mean(xs), T95[len(xs)] * st.stdev(xs) / math.sqrt(len(xs))


def cell_metrics(d):
    e = d["e2e_latencies"]; tt = d["ttfts"]; o = d["output_lens"]
    s = sorted(e); s_t = sorted(tt)
    tpots = [(x - y) / (n - 1) for x, y, n in zip(e, tt, o) if n > 1]
    st_tp = sorted(tpots)
    return {"n": len(e), "e2e_mean_s": st.mean(e), "e2e_p99_idx_s": s[int(0.99 * len(s))], "ttft_mean_s": st.mean(tt),
            "ttft_p99_idx_s": s_t[int(0.99 * len(s_t))], "tpot_p99_idx_s": st_tp[int(0.99 * len(st_tp))], "tpot_mean_recomputed_s": st.mean(tpots),
            "mean_tpot_ms": d["mean_tpot_ms"], "mean_ttft_ms": d["mean_ttft_ms"], "mean_e2e_latency_ms": d["mean_e2e_latency_ms"],
            "p99_e2e_latency_ms": d["p99_e2e_latency_ms"], "p99_ttft_ms": d["p99_ttft_ms"], "p99_tpot_ms": d["p99_tpot_ms"],
            "tokens_mean": st.mean(o), "limit_hits": sum(f == "length" for f in d["finish_reasons"]), "errors": sum(1 for x in d["errors"] if x),
            "completed": d.get("completed"), "duration_s": d.get("duration")}


cells, pairs, problems = [], [], []
scores = {r: json.loads((SCORES / f).read_text()) for r, f in SCORE_FILES.items()}
for ds, (dname, q, rates, suite, metric) in DS.items():
    for rate, mult in rates.items():
        for rep in REPS:
            rec = {}
            for arm in ARMS:
                cdir = ROOT / f"natural_r{rep}" / f"harvest-{arm}-{ds}-{rate}" / "cell"
                res = [p for p in cdir.glob("*_rep*.jsonl") if "arrival" not in p.name and "load" not in p.name]
                if len(res) != 1:
                    problems.append(f"{ds} {rate} {rep} {arm}: {len(res)} result files"); continue
                res = res[0]; stem = res.name[:-6]
                d = json.loads(res.open().readline())
                cmd = json.loads((cdir / f"{stem}.command.json").read_text())
                aud = json.loads((cdir / f"{stem}.audit.json").read_text())
                arr = cdir / f"{stem}.arrival.requests.jsonl"
                key = next((k for k in scores[rep] if not k.startswith("__") and k.endswith(f"natural_r{rep}/harvest-{arm}-{ds}-{rate}/cell/{res.name}")), None)
                sc = scores[rep][key] if key else None
                m = cell_metrics(d)
                c = dict(dataset=dname, q=q, rate=rate, mult=mult, rep=rep, arm=arm, result=str(res.relative_to(ROOT)), result_sha256=sha(res),
                         result_bytes=res.stat().st_size, arrival_sha256=sha(arr), command_status=cmd.get("status"), audit_status=aud.get("status"),
                         submitted=aud.get("submitted"), completed_audit=aud.get("completed"), started_unix_s=cmd.get("started_unix_s"),
                         ended_unix_s=cmd.get("ended_unix_s"), seed=cmd.get("seed"), num_prompts=cmd.get("num_prompts"), workload=cmd.get("workload"),
                         requests_sha256=cmd.get("requests_sha256"), deployment_manifest_sha256=cmd.get("deployment_manifest_sha256"),
                         quality_metric=metric, quality=(sc["scores"][metric] * 100 if sc else None), quality_n=(sc["n"] if sc else None),
                         score_file=SCORE_FILES[rep], **m)
                cells.append(c); rec[arm] = (c, d)
            if len(rec) != 2:
                continue
            (u, ud), (v, vd) = rec["upstream_g1024"], rec["vskipper"]
            pct = {}
            for k_u, k in (("e2e_mean_s", "E2E mean"), ("e2e_p99_idx_s", "E2E p99 (analyze_natural index rule)"), ("ttft_mean_s", "TTFT mean"),
                           ("mean_tpot_ms", "TPOT mean (benchmark mean_tpot_ms)"), ("p99_e2e_latency_ms", "E2E p99 (benchmark field)"),
                           ("p99_ttft_ms", "TTFT p99 (benchmark field)"), ("p99_tpot_ms", "TPOT p99 (benchmark field)"),
                           ("ttft_p99_idx_s", "TTFT p99 (index rule)"), ("tpot_p99_idx_s", "TPOT p99 (index rule)")):
                pct[k] = 100 * (v[k_u] - u[k_u]) / u[k_u]
            pr = dict(dataset=dname, rate=rate, mult=mult, rep=rep, pct=pct,
                      quality_gap_pp=(v["quality"] - u["quality"]) if (u["quality"] is not None and v["quality"] is not None) else None,
                      arrival_byte_identical=u["arrival_sha256"] == v["arrival_sha256"], requests_sha256_equal=u["requests_sha256"] == v["requests_sha256"])
            if mult == "1.25":
                try:
                    pr["capacity_window"] = tps.pair(ud, vd)
                except SystemExit as exc:
                    pr["capacity_window"] = {"refused": str(exc)}
            pairs.append(pr)

summary = []
for ds, (dname, q, rates, suite, metric) in DS.items():
    for rate, mult in rates.items():
        ps = [p for p in pairs if p["dataset"] == dname and p["rate"] == rate]
        row = dict(dataset=dname, q=q, mult=mult, rate=rate, n=len(ps), metrics={})
        for k in ps[0]["pct"]:
            vals = [p["pct"][k] for p in ps]
            mn, h = ci(vals)
            row["metrics"][k] = dict(mean_pct=mn, ci95_half=h, per_rep=vals)
        gaps = [p["quality_gap_pp"] for p in ps]
        mn, h = ci(gaps)
        uq = [c["quality"] for c in cells if c["dataset"] == dname and c["rate"] == rate and c["arm"] == "upstream_g1024"]
        vq = [c["quality"] for c in cells if c["dataset"] == dname and c["rate"] == rate and c["arm"] == "vskipper"]
        row["quality"] = dict(metric=metric, upstream_mean=st.mean(uq), vskipper_mean=st.mean(vq), gap_pp=mn, gap_ci95_half=h, per_rep_gap=gaps)
        if mult == "1.25":
            row["capacity_window_per_rep"] = [p.get("capacity_window") for p in ps]
        summary.append(row)

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "pen_cells.json").write_text(json.dumps(cells, indent=1))
(OUT / "pen_pairs.json").write_text(json.dumps(pairs, indent=1, default=str))
(OUT / "pen_summary.json").write_text(json.dumps(summary, indent=1, default=str))
print(json.dumps(dict(cells=len(cells), pairs=len(pairs), problems=problems)))
