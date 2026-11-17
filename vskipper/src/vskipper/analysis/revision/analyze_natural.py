#!/usr/bin/env python3
"""analyze_natural.py -- natural-stopping cells across repetitions: latency, tokens, limit hits and quality from the same cells (owner 2026-11-06).

Reads <root>/natural_r<K>/harvest-<arm>-<ds>-<label>/cell/<suite>_qps*_rep*.jsonl. Each treatment is paired with upstream inside its repetition, and
the mean of the per-repetition percentage changes carries a t-based 95 % interval, as the paper's Statistics paragraph. Work differs between the
stacks here, so makespan and TPS are not compared. Quality comes from a score file written by score_natural_lane_lmeval.py over the same cells.

usage: analyze_natural.py ROOT DATASET --reps 1,2,3 [--baseline upstream_g1024] [--scores scores.json] [--json out.json]
--reps names one experiment's repetitions: the natural table, the checkpoints and each penalty share natural_r* and must never pool.
"""
import argparse, json, re, statistics as st
from pathlib import Path

T95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571}


def ci(xs):
    n = len(xs)
    return (st.mean(xs), (T95[n] * st.stdev(xs) / n ** 0.5) if n > 1 else None, n)


def cell(path: Path) -> dict:
    d = json.loads(path.open().readline())
    e, o, f = d["e2e_latencies"], d["output_lens"], d["finish_reasons"]
    s = sorted(e)
    return {"n": len(e), "e2e": st.mean(e), "p50": st.median(e), "p99": s[int(0.99 * len(s))], "ttft": st.mean(d["ttfts"]), "tpot": d.get("mean_tpot_ms"),
            "tokens": st.mean(o), "tokens_p50": st.median(o), "limit_hits": sum(x == "length" for x in f), "path": str(path)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("dataset")
    ap.add_argument("--reps", required=True, help="comma-separated natural_r numbers of one experiment")
    ap.add_argument("--baseline", default="upstream_g1024")
    ap.add_argument("--scores", type=Path)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    table = {}
    wanted = set(a.reps.split(","))
    for d in sorted(a.root.glob(f"natural_r*/harvest-*-{a.dataset}-r*")):
        if ".VOID" in d.name or d.parent.name.split("_r")[1] not in wanted:
            continue
        m = re.fullmatch(rf"harvest-(.+)-{re.escape(a.dataset)}-(r[0-9p]+)", d.name)
        files = [p for p in (d / "cell").glob("*_rep*.jsonl") if "arrival" not in p.name and "load" not in p.name]
        if m and len(files) == 1:
            rep = d.parent.name.split("_r")[1]
            table[(m.group(2), rep, m.group(1))] = cell(files[0])
    scores = json.loads(a.scores.read_text()) if a.scores else {}
    out = []
    for lbl in sorted({k[0] for k in table}, key=lambda x: float(x[1:].replace("p", "."))):
        reps = sorted({k[1] for k in table if k[0] == lbl})
        arms = sorted({k[2] for k in table if k[0] == lbl and k[2] != a.baseline})
        base = [table[(lbl, r, a.baseline)] for r in reps if (lbl, r, a.baseline) in table]
        row = {"label": lbl, "baseline": {k: st.mean(c[k] for c in base) for k in ("e2e", "p99", "tokens", "limit_hits")} if base else None, "arms": {}}
        for arm in arms:
            pairs = [(table[(lbl, r, a.baseline)], table[(lbl, r, arm)]) for r in reps if (lbl, r, a.baseline) in table and (lbl, r, arm) in table]
            if not pairs:
                continue
            row["arms"][arm] = {"n": len(pairs), "e2e_s": st.mean(p[1]["e2e"] for p in pairs), "tokens": st.mean(p[1]["tokens"] for p in pairs),
                                "limit_hits": st.mean(p[1]["limit_hits"] for p in pairs)}
            for key in ("e2e", "p99", "ttft", "tpot"):
                row["arms"][arm][f"{key}_pct"] = ci([100 * (t[key] - b[key]) / b[key] for b, t in pairs])
        out.append(row)
        print(lbl, json.dumps(row)[:900])
    if a.json:
        a.json.write_text(json.dumps({"rows": out, "scores": scores}, indent=1))


if __name__ == "__main__":
    main()
