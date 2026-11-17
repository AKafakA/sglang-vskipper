"""Read-only audit of the existing P/D records.

Independent of the campaign's paired_analysis.py: reads every accepted cell's last JSONL record, its command, audit and
work-gate records, pairs upstream and vSkipper by (layout, repetition, rate), compares the two arms' arrival records, and
recomputes the per-repetition % change, the mean and the t-based 95 % half-width (t = 2.571 at n = 6). Writes JSON and TSV.
"""
import hashlib, json, math, os, statistics as st, sys
from pathlib import Path

ROOT = Path(sys.argv[1])            # extracted .../campaign
OUT = Path(sys.argv[2])             # output dir
LAYOUTS = {"pd1p1d": ("1P1D", 14, ["r10p5", "r13p3", "r17p5"]), "pd2p2d": ("2P2D", 26, ["r19p5", "r24p7", "r32p5"])}
MULT = ["0.75", "0.95", "1.25"]
ARMS = ("upstream_pd_g1024", "vskipper_pd")
FIELDS = {"E2E p99": "p99_e2e_latency_ms", "TTFT p99": "p99_ttft_ms", "TPOT p99": "p99_tpot_ms", "makespan": "duration"}
T975 = {6: 2.571}


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def last_record(p: Path) -> dict:
    last = None
    with p.open() as f:
        for line in f:
            if line.strip():
                last = line
    return json.loads(last)


def arrival_rows(p: Path) -> list:
    return [json.loads(l) for l in p.open() if l.strip()]


cells, pairs, problems = [], [], []
for L, (lname, q, rates) in LAYOUTS.items():
    for k in range(1, 7):
        ds = ROOT / f"{L}_r{k}_gsm8k" / f"rep{k}" / "gsm8k"
        for ri, rate in enumerate(rates):
            suite = f"gsm8k_eqw_{rate}"
            gate_p = ds / f"workgate_{suite}.json"
            gate = json.loads(gate_p.read_text()) if gate_p.is_file() else None
            rec = {}
            for arm in ARMS:
                cdir = ds / arm / "cells"
                m = [p for p in cdir.rglob(f"{suite}_qps*_rep*.jsonl") if "arrival" not in p.name and "load" not in p.name]
                if len(m) != 1:
                    problems.append(f"{L} rep{k} {rate} {arm}: {len(m)} result files")
                    continue
                res = m[0]
                stem = res.name[:-len(".jsonl")]
                cmd = json.loads((res.parent / f"{stem}.command.json").read_text())
                aud_p = res.parent / f"{stem}.audit.json"
                aud = json.loads(aud_p.read_text()) if aud_p.is_file() else {}
                arr_p = res.parent / f"{stem}.arrival.requests.jsonl"
                lr = last_record(res)
                a = cmd.get("arrival_schedule", {})
                c = dict(layout=lname, q=q, mult=MULT[ri], rate=rate, rep=k, arm=arm,
                         result=str(res.relative_to(ROOT)), result_sha256=sha(res), result_bytes=res.stat().st_size,
                         command_sha256=sha(res.parent / f"{stem}.command.json"),
                         audit_sha256=sha(aud_p) if aud_p.is_file() else "", arrival_sha256=sha(arr_p) if arr_p.is_file() else "",
                         arrival_bytes=arr_p.stat().st_size if arr_p.is_file() else None,
                         status=cmd.get("status"), started_unix_s=cmd.get("started_unix_s"), ended_unix_s=cmd.get("ended_unix_s"),
                         num_prompts=cmd.get("num_prompts"), seed=cmd.get("seed"), qps=cmd.get("qps"),
                         requests_sha256=cmd.get("requests_sha256"), metadata_sha256=cmd.get("metadata_sha256"),
                         deployment_manifest_sha256=cmd.get("deployment_manifest_sha256"),
                         arr_fields=sorted(a.keys()), arr=a,
                         audit_status=aud.get("status"), submitted=aud.get("submitted"), completed=aud.get("completed"),
                         reported=aud.get("reported"), errors=len(aud.get("errors") or []),
                         completed_lr=lr.get("completed"),
                         **{f: lr.get(v) for f, v in FIELDS.items()},
                         n_start_offsets=len(lr.get("request_start_offsets_s") or []),
                         output_lens_sha256=hashlib.sha256(json.dumps(lr.get("output_lens")).encode()).hexdigest() if lr.get("output_lens") is not None else "",
                         total_output_tokens=sum(lr.get("output_lens") or []) if lr.get("output_lens") is not None else lr.get("total_output_tokens"))
                cells.append(c)
                rec[arm] = (c, arr_p)
            if len(rec) != 2:
                continue
            (u, ua), (v, va) = rec[ARMS[0]], rec[ARMS[1]]
            # arrival comparison: bytes, then the schedule fields, then the per-request content
            byte_identical = u["arrival_sha256"] == v["arrival_sha256"] and u["arrival_sha256"] != ""
            ur, vr = arrival_rows(ua), arrival_rows(va)
            keys_u = sorted(set().union(*[r.keys() for r in ur])) if ur else []
            same_rows = len(ur) == len(vr)
            diff_keys = set()
            for x, y in zip(ur, vr):
                for kk in set(x) | set(y):
                    if x.get(kk) != y.get(kk):
                        diff_keys.add(kk)
            sched_fields_equal = {kk: u["arr"].get(kk) == v["arr"].get(kk) for kk in sorted(set(u["arr"]) | set(v["arr"]))}
            pct = {f: (v[f] - u[f]) / u[f] * 100.0 for f in FIELDS if u.get(f) and v.get(f) is not None}
            pairs.append(dict(layout=lname, mult=MULT[ri], rate=rate, rep=k,
                              gate_pass=(gate or {}).get("pass"), gate_mismatch=(gate or {}).get("mismatch_count"),
                              gate_keys=sorted((gate or {}).keys()), gate_sha256=sha(gate_p) if gate_p.is_file() else "",
                              arrival_byte_identical=byte_identical, arrival_rows=(len(ur), len(vr)), arrival_row_keys=keys_u,
                              arrival_differing_keys=sorted(diff_keys), arrival_schedule_fields_equal=sched_fields_equal,
                              requests_sha256_equal=u["requests_sha256"] == v["requests_sha256"],
                              output_lens_equal=(u["output_lens_sha256"] == v["output_lens_sha256"] and u["output_lens_sha256"] != ""),
                              tokens=(u["total_output_tokens"], v["total_output_tokens"]),
                              completed=(u["completed"], v["completed"]), status=(u["audit_status"], v["audit_status"]),
                              pct=pct))

# recompute mean and CI per (layout, rate, metric)
summary = []
for L, (lname, q, rates) in LAYOUTS.items():
    for ri, rate in enumerate(rates):
        ps = [p for p in pairs if p["layout"] == lname and p["rate"] == rate and p["gate_pass"] is True]
        for f in FIELDS:
            vals = [p["pct"][f] for p in sorted(ps, key=lambda p: p["rep"]) if f in p["pct"]]
            n = len(vals)
            mean = st.mean(vals) if vals else float("nan")
            half = T975[n] * st.stdev(vals) / math.sqrt(n) if n in T975 else float("nan")
            ua = [c[f] for c in cells if c["layout"] == lname and c["rate"] == rate and c["arm"] == ARMS[0]]
            va = [c[f] for c in cells if c["layout"] == lname and c["rate"] == rate and c["arm"] == ARMS[1]]
            summary.append(dict(layout=lname, q=q, mult=MULT[ri], rate=rate, metric=f, n=n, per_rep=vals, mean_pct=mean,
                                ci95_half=half, upstream_abs=st.fmean(ua), vskipper_abs=st.fmean(va)))

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "pd_audit_cells.json").write_text(json.dumps(cells, indent=1, default=str))
(OUT / "pd_audit_pairs.json").write_text(json.dumps(pairs, indent=1, default=str))
(OUT / "pd_recompute_24.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(dict(cells=len(cells), pairs=len(pairs), problems=problems)))
