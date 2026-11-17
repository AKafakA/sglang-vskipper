# read-only: served decode skip of the released checkpoint (arm vskipper) in the matched-work GSM8K ablation cells (Q* 11:
# r8p25 / r10p45 / r13p75 = 0.75 / 0.95 / 1.25 x Q*; units e3p4_r1, e3p2_r2, e3p2_r3, the nine that e3_read.sh pools for vskipper). Same definitions and counters as served_loops_skip.py, from
# server_info.after minus server_info.before: skip over routed rows = project_rows / layer_rows (the server's skip_ratio, undefined (None) when
# no decode row ran in routed mode), skip over all decoded tokens = project_rows / (16 x sum(output_len - 1)), routed share of decode rows =
# layer_rows / (16 x sum(output_len - 1)), and the all-token skip over the server's own decode-row count.
# usage: e3_matched_skip.py EXTRACT_ROOT OUT.json
import json, sys
ROOT, OUT = sys.argv[1], sys.argv[2]
UNITS = {1: "e3p4_r1", 2: "e3p2_r2", 3: "e3p2_r3"}
RATES = ("r8p25", "r10p45", "r13p75")
def counters(path):                                                             # fail closed: a missing counter stops the script
    vr = json.load(open(path))["internal_states"][0]["vp_runtime"]
    dec = vr["model"]["flexidepth"]["full_graph_routes"]["by_phase"]["decode"]; c3 = vr["fd_c3"]["counters"]
    return dec["layer_rows"], dec["project_rows"], c3["fd_tokens_skip_body"] + c3["fd_tokens_prod_allrun_band"] + c3["fd_tokens_dense_overflow"]
res = {}
for rate in RATES:
    for rep, unit in UNITS.items():
        base = f"{ROOT}/{unit}_{rate}_gsm8k/rep{rep}/gsm8k/vskipper/cells/gsm8k_eqw_{rate}_{rate.replace('r', 'qps', 1)}_rep1"
        lines = open(base + ".jsonl").read().splitlines()
        assert len(lines) == 1, (base, len(lines))
        c = json.loads(lines[0]); n = c["output_lens"]
        assert c["submitted"] == c["completed"] == len(n) == 3600, (base, c["submitted"], c["completed"], len(n))
        assert c["errors"] in (0, [], None) or not any(c["errors"]), (base, "errors")
        la, pa, sa = counters(base + ".server_info.after.json"); lb, pb, sb = counters(base + ".server_info.before.json")
        lr, pr, srv = la - lb, pa - pb, sa - sb; dtok = sum(x - 1 for x in n if x > 0)
        res[f"vskipper|{rate}|{rep}"] = {
            "unit": unit, "requests": len(n), "output_tokens": sum(n), "finish_length": c["finish_reasons"].count("length"),
            "decode_layer_rows": lr, "decode_project_rows": pr, "skip_conditional": pr / lr if lr else None,
            "skip_all_token": pr / (16 * dtok), "routed_share_of_decode_rows": lr / (16 * dtok),
            "server_decode_rows": srv, "skip_all_token_server_rows": pr / (16 * srv), "before_layer_rows": lb}
json.dump(res, open(OUT, "w"), indent=1)
for rate in RATES:
    rows = [res[f"vskipper|{rate}|{k}"] for k in UNITS]
    print(f"{rate:7s} released matched work | tokens {[r['output_tokens'] for r in rows]} | skip cond "
          f"{[None if r['skip_conditional'] is None else round(r['skip_conditional'], 4) for r in rows]} all-token "
          f"{[round(r['skip_all_token'], 4) for r in rows]} routed share {[round(r['routed_share_of_decode_rows'], 3) for r in rows]} "
          f"all-token over server rows {[round(r['skip_all_token_server_rows'], 4) for r in rows]} before-rows {[r['before_layer_rows'] for r in rows]}")
