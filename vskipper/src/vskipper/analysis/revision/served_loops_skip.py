# read-only: per served cell, repetition by the dev evaluator's two detectors (fdfix_lib.is_loop and Antidoom's find_inner_repetition with REP_CFG,
# both taken verbatim from fdfix_lib.py without importing torch), output-cap hits, mean output tokens, and decode skip: conditional
# (project_rows / layer_rows over routed decode rows, the server's skip_ratio) and all-token (project_rows / (routed layers x all decode tokens)),
# from server_info.after minus server_info.before (and, for comparison, all-token over the server's own decode-row count).   usage: served_loops_skip.py CAMPAIGN_ROOT REP0 CAND_KEY FDFIX_LIB OUT.json
import glob, json, os, re, statistics as st, sys
ROOT, REP0, KEY, LIB, OUT = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]
src = open(LIB).read()
a = src.index("def is_loop(text):"); b = src.index("@torch.no_grad()", a)
ns = {"re": re}; exec(compile(src[a:b], LIB, "exec"), ns)                     # is_loop, REP_CFG, _verify_repetition_at, find_inner_repetition
is_loop, fir, CFG = ns["is_loop"], ns["find_inner_repetition"], ns["REP_CFG"]
ARMS = {"upstream": ("upstream_g1024", "gsm8k"), "released": ("vskipper", "gsm8k"), "candidate": ("vskipper_cand_grpo", KEY)}
def counters(path):                                                             # fail closed: a missing counter stops the script
    vr = json.load(open(path))["internal_states"][0]["vp_runtime"]
    dec = vr["model"]["flexidepth"]["full_graph_routes"]["by_phase"]["decode"]; c3 = vr["fd_c3"]["counters"]
    return dec["layer_rows"], dec["project_rows"], c3["fd_tokens_skip_body"] + c3["fd_tokens_prod_allrun_band"] + c3["fd_tokens_dense_overflow"]
res = {}
for rate in ("r8p25", "r10p45", "r13p75"):
    for k in (1, 2, 3):
        rep = REP0 + k
        for name, (arm, ds) in ARMS.items():
            d = f"{ROOT}/natural_r{rep}/harvest-{arm}-{ds}-{rate}"
            fs = [f for f in glob.glob(f"{d}/cell/*_rep*.jsonl") if "arrival" not in os.path.basename(f) and ".load" not in os.path.basename(f)
                  and not os.path.basename(f).startswith("INVALID")]
            assert len(fs) == 1, (d, fs)
            c = json.loads(open(fs[0]).read().splitlines()[-1])
            txt, n = c["generated_texts"], c["output_lens"]
            assert len(txt) == 3600 == len(n)
            line = sum(1 for t in txt if is_loop(t)); anti = sum(1 for t in txt if fir(t, **CFG) is not None)
            row = {"requests": len(n), "mean_output_tokens": st.mean(n), "cap_hits": c["finish_reasons"].count("length"),
                   "loop_line": line, "loop_antidoom": anti, "loop_either": sum(1 for t in txt if is_loop(t) or fir(t, **CFG) is not None)}
            if name != "upstream":
                la, pa, sa = counters(f"{d}/server_info.after.json"); lb, pb, sb = counters(f"{d}/server_info.before.json")
                lr, pr, srv = la - lb, pa - pb, sa - sb; dtok = sum(x - 1 for x in n if x > 0)
                # skip over routed rows is undefined (None) when no decode row ran in routed mode
                row |= {"decode_layer_rows": lr, "decode_project_rows": pr, "skip_conditional": pr / lr if lr else None,
                        "skip_all_token": pr / (16 * dtok), "routed_share_of_decode_rows": lr / (16 * dtok),
                        "server_decode_rows": srv, "skip_all_token_server_rows": pr / (16 * srv), "before_layer_rows": lb}
            res[f"{name}|{rate}|{rep}"] = row
json.dump(res, open(OUT, "w"), indent=1)
for rate in ("r8p25", "r10p45", "r13p75"):
    for name in ARMS:
        rows = [res[f"{name}|{rate}|{REP0 + k}"] for k in (1, 2, 3)]
        s = f"{rate:7s} {name:9s} tokens {st.mean(r['mean_output_tokens'] for r in rows):6.1f} | cap {[r['cap_hits'] for r in rows]} | loop line {[r['loop_line'] for r in rows]} antidoom {[r['loop_antidoom'] for r in rows]} either {[r['loop_either'] for r in rows]}"
        if name != "upstream":
            s += f" | skip cond {[None if r['skip_conditional'] is None else round(r['skip_conditional'], 4) for r in rows]} all-token {[round(r['skip_all_token'], 4) for r in rows]} routed share {[round(r['routed_share_of_decode_rows'], 3) for r in rows]} before-rows {[r['before_layer_rows'] for r in rows]}"
        print(s)
