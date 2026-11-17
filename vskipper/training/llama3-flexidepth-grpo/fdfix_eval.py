#!/usr/bin/env python3
"""fdfix_eval.py -- the one held-out evaluator for every candidate (released, GRPO checkpoints), so results are comparable.

Dev rows only (pool.json dev_indices[:dev_limit]); prompts = the salt-1 dev prompts of fdfix_grpo.py (demonstrations from the training pool).
Greedy, dense prompt and routed decode, max_new 1024, no early stop. Batches are fixed chunks of --gen-micro rows in dev order, dealt
round-robin to ranks, so a candidate's batch composition does not depend on the GPU count.
Candidate = --model directory and, optionally, --router-pt replacing router.* and router_proj.* (a GRPO export).
Per row: strict, flexible, line-repeat loop, Antidoom-detector loop, cap hit (no end-of-turn within 1024), tokens, routed skip fraction over
the decisions decode executed (recorded during generation: every completion token fed back, i.e. all but the last), layers used, and as a
cross-check the skip fraction of a teacher-forced replay of the same tokens (the GRPO reward's definition). With --baseline DIR: paired
per-row differences against that evaluation with a paired bootstrap 95% interval (10,000 resamples, fixed seed).
"""
import argparse
import json
import os
import random
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fdfix_lib as L  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--router-pt", default=""); ap.add_argument("--released-pt", required=True)
ap.add_argument("--routed-prompt", action="store_true", help="route the prompt natively, as the served FD prefill (default: dense prompt)")
ap.add_argument("--dense-decode", action="store_true", help="force every decode step dense: the base model's layers (the served stock decode)")
ap.add_argument("--rows", required=True); ap.add_argument("--pool", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--dev-limit", type=int, default=500); ap.add_argument("--gen-micro", type=int, default=32); ap.add_argument("--max-new", type=int, default=1024)
ap.add_argument("--baseline", default=""); ap.add_argument("--label", required=True)
ap.add_argument("--loop-stop", action="store_true", help="stop a row once Antidoom's repetition detector fires (checked every 32 tokens)")
ap.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16", help="weights and compute dtype (the served profile is fp16)")
ap.add_argument("--tokenizer", default="", help="tokenizer and chat template directory (default: --model); every candidate uses the released one")
a = ap.parse_args()
assert not (a.routed_prompt and a.dense_decode), "one served regime per run"
T0 = time.time()
L.AMP = {"bf16": torch.bfloat16, "fp16": torch.float16}[a.dtype]
D = L.Dist()
assert not os.path.exists(os.path.join(a.out, "summary.json")), f"{a.out} already evaluated: never overwritten"
if D.main:
    os.makedirs(a.out, exist_ok=True)
D.barrier()
rows, TRAIN_IDX, DEV_IDX = L.load_rows(a.rows, a.pool, need_dev=True, dev_limit=a.dev_limit)
tok, model = L.load_model(a.model, D.local)
if a.tokenizer:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
R = L.Routing(model)
P = L.Prompts(tok, rows, TRAIN_IDX)
params = dict(model.named_parameters())
rp_keys = sorted(k for k in params if ".router." in k or ".router_proj." in k)
ident = {"label": a.label, "model_id": L.model_identity(a.model), "tokenizer_dir": a.tokenizer or a.model, "router_pt": a.router_pt,
         "router_pt_sha256": L.sha_file(a.router_pt)[:16] if a.router_pt else None}
if a.router_pt:
    new = torch.load(a.router_pt, map_location="cpu")
    assert set(new) == set(rp_keys), "router file keys differ from the model's router/projector tensors"
    with torch.no_grad():
        for k in rp_keys:
            assert new[k].shape == params[k].shape, k
            params[k].copy_(new[k].to(params[k].device, params[k].dtype))           # bf16 file into a bf16 or fp16 model
ident["dtype"] = a.dtype; ident["loop_stop"] = a.loop_stop; ident["prompt_routed"] = a.routed_prompt; ident["decode_dense"] = a.dense_decode
released = torch.load(a.released_pt, map_location="cpu")
ident["router_projector_equal_released"] = all(torch.equal(params[k].detach().float().cpu(), released[k].float()) for k in rp_keys)
dev = DEV_IDX[:a.dev_limit]
batches = [dev[s:s + a.gen_micro] for s in range(0, len(dev), a.gen_micro)]
eot = tok.convert_tokens_to_ids("<|eot_id|>")
rec = []
for b in batches[D.rank::D.world]:
    gens = L.generate(model, R, tok, [P.ids(i, 1) for i in b], a.max_new, D.dev, sample=False, loop_stop=a.loop_stop,
                      prompt_routed=a.routed_prompt, decode_dense=a.dense_decode)
    for s in range(0, len(b), 4):
        part = gens[s:s + 4]; qs = b[s:s + 4]
        seqs = [P.ids(i, 1) + g["completion"] for i, g in zip(qs, part)]
        dl = [len(x) for x in seqs] if a.dense_decode else [0] * len(qs) if a.routed_prompt else [len(P.ids(i, 1)) for i in qs]
        _, masks, _ = L.teacher_forced(model, R, seqs, dl, D.dev); R.reset()             # the replay in the same regime
        for r, (i, g) in enumerate(zip(qs, part)):
            pl, n = len(P.ids(i, 1)), len(g["completion"])
            dec = masks[r, pl:pl + n - 1] if n >= 2 else None                  # [n-1, routed]
            replay = float(1 - dec.mean()) if dec is not None else 0.0
            skip = g["exec_skip"]                                              # the decisions decode executed
            text = tok.decode(g["completion"], skip_special_tokens=True)
            st, fl = L.score(text, P.gold(i))
            rec.append({"dev_index": i, "strict": st, "flexible": fl, "loop": L.is_loop(text),
                        "loop_antidoom": L.find_inner_repetition(text, **L.REP_CFG) is not None, "cap_hit": not g["ended"] and not g["loop_stopped"],
                        "tokens": n, "skip_fraction": round(skip, 5), "layers_used": round(R.n_layers - len(R.layers) * skip, 4),
                        "skip_fraction_replay": round(replay, 5), "loop_stopped": g["loop_stopped"],
                        "completion": g["completion"]})
allr = sorted([x for part in D.gather(rec) for x in part], key=lambda z: z["dev_index"])
if D.main:
    assert [x["dev_index"] for x in allr] == sorted(dev), "missing dev rows"
    with open(os.path.join(a.out, "rows.jsonl"), "w") as f:
        for x in allr:
            f.write(json.dumps(x) + "\n")
    n = len(allr)
    KEYS = ("strict", "flexible", "loop", "loop_antidoom", "cap_hit", "tokens", "skip_fraction", "layers_used", "skip_fraction_replay")
    summ = ident | {"n": n, "seconds": round(time.time() - T0, 1), "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)} | \
        {k: round(sum(float(x[k]) for x in allr) / n, 4) for k in KEYS}
    if a.baseline:
        base = {json.loads(l)["dev_index"]: json.loads(l) for l in open(os.path.join(a.baseline, "rows.jsonl"))}
        assert set(base) == {x["dev_index"] for x in allr}, "baseline rows differ"
        rng = random.Random(20261114)
        paired = {}
        for k in KEYS:
            d = [float(x[k]) - float(base[x["dev_index"]][k]) for x in allr]
            m = sum(d) / n
            boots = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(10000))
            paired[k] = {"mean_diff": round(m, 5), "ci95": [round(boots[249], 5), round(boots[9750], 5)]}
        paired["identical_completions"] = sum(x["completion"] == base[x["dev_index"]]["completion"] for x in allr)
        summ["paired_vs"] = {"baseline": a.baseline, "baseline_loop_stop": json.load(open(os.path.join(a.baseline, "summary.json"))).get("loop_stop", False),
                             **paired}
    json.dump(summ, open(os.path.join(a.out, "summary.json.tmp"), "w"), indent=1)            # summary.json marks completion: written whole
    os.replace(os.path.join(a.out, "summary.json.tmp"), os.path.join(a.out, "summary.json"))
    print("EVAL", json.dumps(summ), flush=True)
D.close()
