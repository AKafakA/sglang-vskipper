#!/usr/bin/env python3
"""tps_audit.py MANIFEST.json OUT.json -- one counting analysis for every TPS/RPS row (read-only, no runs).

For each paired repetition (upstream cell U, treatment cell V, both the last JSON line of a benchmark cell):
  window      a = first arrival of either arm, b = upstream's last arrival (capacity_window.py's window); the two arms' last
              arrivals must agree within 1.0 s and both arms must complete fewer requests than arrive in [a, b] (saturation).
  token time  request start + TTFT for the first token, then + each inter-token latency. A request whose output_lens exceeds its
              timed tokens has untimed tokens. They are never given an invented time: each lies somewhere in its request's valid
              context [first-token time, request end]. Lower bound counts it only if that interval lies inside the window, upper
              bound if the interval overlaps the window.
  TPS         EXACT: timed tokens with a <= t <= b, divided by (b - a).
              BIN:   timed tokens whose 50 ms bin index lies in [floor(a/0.05), floor(b/0.05)] (capacity_window.py's rule).
              Untimed tokens: the same two rules applied to the bounds of their valid interval.
  RPS         completions with a <= end <= b, divided by (b - a) (the same under both rules).
  change      per repetition 100 (V/U - 1). TPS bounds: lower = V_low/U_high, upper = V_high/U_low.
  row         mean over repetitions and the t-based 95 % half-width (capacity_window.py's T975 by n - 1).
The bin-boundary effect is EXACT(timed) - BIN(timed). The missing-time effect is the width of the bounds.
"""
import hashlib, json, math, statistics as st, sys

BIN = 0.05
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365}


def load(path):
    with open(path, "rb") as f:
        raw = f.read()
    return json.loads(raw.decode().strip().splitlines()[-1]), hashlib.sha256(raw).hexdigest()


def counts(c, a, b):
    lo, hi = math.floor(a / BIN), math.floor(b / BIN)
    r = {"exact": 0, "bin": 0, "untimed": 0, "unt_exact_sure": 0, "unt_exact_maybe": 0, "unt_bin_sure": 0, "unt_bin_maybe": 0}
    for s, ttft, itls, ok, n, e in zip(c["request_start_offsets_s"], c["ttfts"], c["itls"], c["successes"], c["output_lens"],
                                       c["request_end_offsets_s"]):
        if not ok:
            raise SystemExit("failed request in a cell")
        if n == 0:
            continue
        t = s + ttft
        times = [t]
        for g in itls:
            t += g
            times.append(t)
        r["exact"] += sum(1 for x in times if a <= x <= b)
        r["bin"] += sum(1 for x in times if lo <= math.floor(x / BIN) <= hi)
        m = n - len(times)
        if m < 0:
            raise SystemExit(f"more token times than output tokens ({len(times)} > {n})")
        if m:
            r["untimed"] += m
            t0 = s + ttft
            if a <= t0 and e <= b:
                r["unt_exact_sure"] += m
            elif not (e < a or t0 > b):
                r["unt_exact_maybe"] += m
            b0, b1 = math.floor(t0 / BIN), math.floor(e / BIN)
            if lo <= b0 and b1 <= hi:
                r["unt_bin_sure"] += m
            elif not (b1 < lo or b0 > hi):
                r["unt_bin_maybe"] += m
    return r


def pair(u, v):
    su, sv = u["request_start_offsets_s"], v["request_start_offsets_s"]
    a, b = min(su + sv), max(su)
    w = b - a
    out = {"a": a, "b": b, "window_s": w, "last_arrival_diff_s": abs(b - max(sv))}
    for tag, c in (("U", u), ("V", v)):
        k = counts(c, a, b)
        done = sum(1 for x in c["request_end_offsets_s"] if a <= x <= b)
        arrived = sum(1 for x in c["request_start_offsets_s"] if a <= x <= b)
        out[tag] = {**k, "done": done, "arrived": arrived, "done_over_arrived": done / arrived,
                    "requests": len(c["output_lens"]), "output_tokens": sum(c["output_lens"])}
    U, V = out["U"], out["V"]
    def pct(x, y): return 100.0 * (x / y - 1.0)
    for rule in ("exact", "bin"):
        out[f"tps_{rule}_timed"] = pct(V[rule], U[rule])
        u_lo, u_hi = U[rule] + U[f"unt_{rule}_sure"], U[rule] + U[f"unt_{rule}_sure"] + U[f"unt_{rule}_maybe"]
        v_lo, v_hi = V[rule] + V[f"unt_{rule}_sure"], V[rule] + V[f"unt_{rule}_sure"] + V[f"unt_{rule}_maybe"]
        out[f"tps_{rule}_bounds"] = [pct(v_lo, u_hi), pct(v_hi, u_lo)]
    out["rps"] = pct(V["done"], U["done"])
    out["checks"] = {"windows_agree_1s": out["last_arrival_diff_s"] <= 1.0,
                     "saturated": U["done_over_arrived"] < 1 and V["done_over_arrived"] < 1}
    return out


def ci(xs):
    n = len(xs)
    return [st.mean(xs), T975[n - 1] * st.stdev(xs) / math.sqrt(n)] if n > 1 else [xs[0], None]


def main():
    man = json.load(open(sys.argv[1]))
    res = []
    for row in man["rows"]:
        reps = []
        for rp in row["reps"]:
            u, hu = load(rp["upstream"]); v, hv = load(rp["treatment"])
            p = pair(u, v); p["rep"] = rp["rep"]; p["sha256"] = {"upstream": hu, "treatment": hv}
            reps.append(p)
        summ = {"n": len(reps)}
        for key in ("tps_exact_timed", "tps_bin_timed", "rps"):
            summ[key] = ci([r[key] for r in reps])
        for rule in ("exact", "bin"):
            lo = ci([r[f"tps_{rule}_bounds"][0] for r in reps]); hi = ci([r[f"tps_{rule}_bounds"][1] for r in reps])
            summ[f"tps_{rule}_bounds_mean"] = [lo[0], hi[0]]; summ[f"tps_{rule}_bounds_ci"] = [lo[1], hi[1]]
        summ["untimed_tokens_max_per_cell"] = max(max(r["U"]["untimed"], r["V"]["untimed"]) for r in reps)
        summ["all_checks_pass"] = all(all(r["checks"].values()) for r in reps)
        res.append({"row": row["row"], "where": row.get("where"), "printed": row.get("printed"), "summary": summ, "reps": reps})
        s = summ
        print(f"{row['row']:34s} n={s['n']} | TPS exact {s['tps_exact_timed'][0]:+.4f} +- {s['tps_exact_timed'][1]:.4f}"
              f" | TPS bin {s['tps_bin_timed'][0]:+.4f} +- {s['tps_bin_timed'][1]:.4f}"
              f" | RPS {s['rps'][0]:+.4f} +- {s['rps'][1]:.4f} | untimed max {s['untimed_tokens_max_per_cell']} | checks {s['all_checks_pass']}")
    json.dump({"method": __doc__, "rows": res}, open(sys.argv[2], "w"), indent=1)


if __name__ == "__main__":
    main()
