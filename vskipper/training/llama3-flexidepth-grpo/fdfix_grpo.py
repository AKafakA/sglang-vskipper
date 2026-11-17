#!/usr/bin/env python3
"""fdfix_grpo.py -- router+projector GRPO for FlexiDepth-Llama-3-8B-Instruct.

Policy: the checkpoint's own forward with a dense prompt and routed decode (generation: the multi-token prefill is forced dense; the trainer's
teacher-forced forward: prompt positions forced dense by an explicit mask). Forcing = router logit 30 (sigmoid -> 1.0, mask 1, weight 1), which
is exactly the base layer, and passes no gradient to the router at those positions.
One behaviour policy: every rollout is sampled at temperature T; old and reference log-probs are recomputed by the trainer at the same T.
Reference: the released router/projector tensors, swapped in for a no-grad forward. Trainable: router + router_proj (209,784,832).
Data: GSM8K training-pool rows only (pool.json train_indices); dev rows only for evaluation; the test split is never read.
Checkpoints at iteration boundaries only (after the update, before the next rollouts): trainable fp32 tensors, AdamW state, iteration,
global step, RNG states, config hash and reference identity. Sampling RNG is reseeded from (seed, iteration, rank), so a resume replays it.
Launch: torchrun --nproc_per_node N fdfix_grpo.py ...   (or plain python for one GPU)
"""
import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
import time

import torch
import torch.distributed as dist

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--rows", required=True, help="dir with train.jsonl (openai/gsm8k main)")
ap.add_argument("--pool", required=True); ap.add_argument("--released-pt", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--resume", default=""); ap.add_argument("--iterations", type=int, default=24)
ap.add_argument("--prompts", type=int, default=32); ap.add_argument("--group", type=int, default=8)
ap.add_argument("--temperature", type=float, default=0.8); ap.add_argument("--max-new", type=int, default=1024)
ap.add_argument("--gen-micro", type=int, default=16); ap.add_argument("--train-micro", type=int, default=2)
ap.add_argument("--mini-batches", type=int, default=2, help="optimizer steps per iteration")
ap.add_argument("--lr", type=float, default=5e-6); ap.add_argument("--beta", type=float, default=0.04); ap.add_argument("--clip", type=float, default=0.2)
ap.add_argument("--w-correct", type=float, default=1.0); ap.add_argument("--w-loop", type=float, default=1.0); ap.add_argument("--w-cap", type=float, default=0.5)
ap.add_argument("--w-len", type=float, default=0.001); ap.add_argument("--len-ref", type=int, default=256); ap.add_argument("--w-skip", type=float, default=0.2)
ap.add_argument("--w-len-group", type=float, default=0.0,
                help="group-relative length term: -w x (tokens / mean tokens of the prompt's group - 1); no external threshold")
ap.add_argument("--w-skip-ret", type=float, default=0.0,
                help="skip retention, reward term: -w x max(0, the released router's executed skip on the same sample - the sample's executed skip)")
ap.add_argument("--lambda-route", type=float, default=0.0,
                help="skip retention, router loss: lambda x mean over decode positions and routed layers of relu(router weight - released router weight)")
ap.add_argument("--grad-ckpt", action="store_true"); ap.add_argument("--ckpt-every", type=int, default=6)
ap.add_argument("--eval-at", default="", help="comma list of iterations for dev evaluation, 0 = before training")
ap.add_argument("--dev-limit", type=int, default=500); ap.add_argument("--time-budget-min", type=float, default=0)
ap.add_argument("--seed", type=int, default=20261114)
ap.add_argument("--focus-indices", default="", help="JSON list of training-pool indices to oversample (training pool only)")
ap.add_argument("--focus-frac", type=float, default=0.0, help="share of each iteration's prompts drawn from --focus-indices")
ap.add_argument("--allow-cfg-change", default="", help="comma list of config keys a resume may change; each change is logged into the history")
a = ap.parse_args()
T0 = time.time()

WORLD = int(os.environ.get("WORLD_SIZE", "1")); RANK = int(os.environ.get("RANK", "0")); LOCAL = int(os.environ.get("LOCAL_RANK", "0"))
if WORLD > 1:
    dist.init_process_group("nccl")
torch.cuda.set_device(LOCAL)
DEV = torch.device("cuda", LOCAL)
MAIN = RANK == 0
CFG_KEYS = ["prompts", "group", "temperature", "max_new", "gen_micro", "lr", "beta", "clip", "w_correct", "w_loop",
            "w_cap", "w_len", "len_ref", "w_skip", "mini_batches", "seed"]


def log(*x):
    if MAIN:
        print(f"[{time.strftime('%H:%M:%SZ', time.gmtime())}]", *x, flush=True)


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def model_identity(d):
    """Content identity of the model directory's code, config, index and tokenizer (shards are hash-checked by the launcher)."""
    h = hashlib.sha256()
    names = ["config.json", "generation_config.json", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json",
             "special_tokens_map.json"] + sorted(f for f in os.listdir(d) if f.endswith(".py"))
    for n in names:
        p = os.path.join(d, n)
        h.update(n.encode() + (sha_file(p).encode() if os.path.exists(p) else b"absent"))
    return h.hexdigest()[:16]


# identities by content, not by path, so a resume on another host with other paths is accepted and a changed file is refused
CFG = {k: getattr(a, k) for k in CFG_KEYS} | {"world": WORLD, "model_id": model_identity(a.model),
                                             "train_jsonl_sha256": sha_file(os.path.join(a.rows, "train.jsonl"))[:16],
                                             "pool_json_sha256": sha_file(a.pool)[:16]}
if a.focus_indices or a.focus_frac:                                            # the configuration identity is unchanged without them
    CFG |= {"focus_sha256": sha_file(a.focus_indices)[:16] if a.focus_indices else "", "focus_frac": a.focus_frac}
if a.w_len_group:                                                              # likewise: absent unless set
    CFG |= {"w_len_group": a.w_len_group}
if a.w_skip_ret:                                                               # skip retention (owner 11-15): absent unless set
    CFG |= {"w_skip_ret": a.w_skip_ret}
if a.lambda_route:
    CFG |= {"lambda_route": a.lambda_route}
CFG_HASH = hashlib.sha256(json.dumps(CFG, sort_keys=True).encode()).hexdigest()[:16]
if MAIN:
    os.makedirs(a.out, exist_ok=True)
if WORLD > 1:
    dist.barrier()

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, trust_remote_code=True, dtype=torch.bfloat16, use_cache=False, device_map={"": LOCAL})
eot = tok.convert_tokens_to_ids("<|eot_id|>")
routed = list(model.config.routing_layers); NLAYER = model.config.num_hidden_layers
params = dict(model.named_parameters())
TRAIN_KEYS = [n for n in params if ".router." in n or ".router_proj." in n]
ROUTER_KEYS = [n for n in TRAIN_KEYS if ".router." in n]; PROJ_KEYS = [n for n in TRAIN_KEYS if ".router_proj." in n]
FROZEN = [n for n in params if n not in TRAIN_KEYS]
n_train = sum(params[n].numel() for n in TRAIN_KEYS); n_frozen = sum(params[n].numel() for n in FROZEN)
assert n_train == 209_784_832 and n_frozen == 8_030_261_248, (n_train, n_frozen)
for n, p in params.items():
    p.requires_grad_(n in TRAIN_KEYS)
released = torch.load(a.released_pt, map_location="cpu")
assert set(released) == set(TRAIN_KEYS) and all(torch.equal(released[k], params[k].detach().cpu()) for k in TRAIN_KEYS), "model != released router file"
REF_ID = {"released_pt_sha256": sha_file(a.released_pt)[:16]}
REF = {k: v.to(DEV) for k, v in released.items()}
for n in TRAIN_KEYS:
    params[n].data = params[n].data.float()
if a.grad_ckpt:
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})


@torch.no_grad()
def fingerprint(t, chunk=1 << 26):
    v = t.detach().reshape(-1).view(torch.int16)
    s1 = torch.zeros((), dtype=torch.int64, device=v.device); s2 = torch.zeros((), dtype=torch.int64, device=v.device)
    for i in range(0, v.numel(), chunk):
        x = v[i:i + chunk].to(torch.int64)
        s1 += x.sum(); s2 += (x * (torch.arange(i, i + x.numel(), device=v.device) % 1009 + 1)).sum()
    return int(s1), int(s2)


FROZEN_FP = {n: fingerprint(params[n]) for n in FROZEN}

# ---- routing control: dense prompt, routed decode ----
FORCE = {"mode": "native", "mask": None}


def wrap_router(mod):
    orig = mod.forward

    def fwd(h):
        logits = orig(h)
        if FORCE["mode"] == "gen" and h.shape[1] > 1:                       # generation prefill = exactly the prompt
            return torch.full_like(logits, 30.0)
        if FORCE["mode"] == "train" and FORCE["mask"] is not None:          # teacher forcing: explicit prompt mask
            return torch.where(FORCE["mask"][..., None], torch.full_like(logits, 30.0), logits)
        return logits
    mod.forward = fwd


for i in routed:
    wrap_router(model.model.layers[i].router)

# ---- data: training-pool rows only; dev rows are parsed only when a dev evaluation is requested ----
pool = json.load(open(a.pool))
TRAIN_IDX, DEV_IDX = pool["train_indices"], pool["dev_indices"]
assert not set(TRAIN_IDX) & set(DEV_IDX)
EVAL_AT = {int(x) for x in a.eval_at.split(",") if x != ""}
NEED = set(TRAIN_IDX) | (set(DEV_IDX[:a.dev_limit]) if EVAL_AT else set())
rows = {}
with open(os.path.join(a.rows, "train.jsonl")) as f:
    for i, line in enumerate(f):
        if i in NEED:
            rows[i] = json.loads(line)
assert len(rows) == len(NEED), (len(rows), len(NEED))


def gold(i):
    return norm_ans(rows[i]["answer"].split("####")[-1])


def norm_ans(s):
    return re.sub(r"\.$", "", re.sub(r"[,$]", "", s.strip()))


def messages(i, rng):
    demos = rng.sample([j for j in TRAIN_IDX if j != i], 5)          # demonstrations only from the training pool
    m = []
    for j in demos:
        m += [{"role": "user", "content": f"Question: {rows[j]['question'].strip()}\nAnswer:"}, {"role": "assistant", "content": rows[j]["answer"]}]
    return m + [{"role": "user", "content": f"Question: {rows[i]['question'].strip()}\nAnswer:"}]


def prompt_ids(i, salt):
    return tok.apply_chat_template(messages(i, random.Random(a.seed * 7919 + i * 31 + salt)), add_generation_prompt=True, tokenize=True)


STRICT = re.compile(r"#### (\-?[0-9\.\,]+)"); FLEX = re.compile(r"(-?[$0-9.,]{2,})|(-?[0-9]+)")


def score(text, i):
    s = STRICT.search(text); f = FLEX.findall(text)
    strict = s is not None and norm_ans(s.group(1)) == gold(i)
    flex = bool(f) and norm_ans([x for x in f[-1] if x][0]) == gold(i)
    return strict, flex


def is_loop(text):
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    run = 1
    for x, y in zip(lines, lines[1:]):
        run = run + 1 if x == y else 1
        if run >= 3:
            return True
    if len(text) >= 2000:
        w = text[-180:-120]
        if len(w) >= 60 and text.count(w) > 5:
            return True
    return False


# ---- generation and teacher-forced log-probs ----
@torch.no_grad()
def generate(prompts, sample, seed):
    model.eval(); FORCE["mode"] = "gen"
    L = max(map(len, prompts))
    ids = torch.full((len(prompts), L), eot, dtype=torch.long); att = torch.zeros_like(ids)
    for r, p in enumerate(prompts):
        ids[r, L - len(p):] = torch.tensor(p); att[r, L - len(p):] = 1
    torch.manual_seed(seed)
    kw = dict(do_sample=True, temperature=a.temperature, top_p=1.0, top_k=0) if sample else dict(do_sample=False)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model.generate(input_ids=ids.to(DEV), attention_mask=att.to(DEV), max_new_tokens=a.max_new, use_cache=True,
                             pad_token_id=eot, eos_token_id=eot, **kw)
    FORCE["mode"] = "native"
    res = []
    for r, p in enumerate(prompts):
        new = out[r, L:].tolist()
        ended = eot in new
        comp = new[:new.index(eot) + 1] if ended else new
        res.append({"prompt": p, "completion": comp, "ended": ended})
    return res


def batch_tensors(items):
    L = max(len(x["prompt"]) + len(x["completion"]) for x in items)
    ids = torch.full((len(items), L), eot, dtype=torch.long); att = torch.zeros_like(ids)
    pm = torch.zeros((len(items), L), dtype=torch.bool); cm = torch.zeros((len(items), L), dtype=torch.bool); dm = torch.zeros_like(cm)
    for r, x in enumerate(items):
        s = x["prompt"] + x["completion"]; P = len(x["prompt"])
        ids[r, :len(s)] = torch.tensor(s); att[r, :len(s)] = 1; pm[r, :P] = True; cm[r, P:len(s)] = True
        dm[r, P:len(s) - 1] = True      # routed decisions generation executed: each completion token fed back, i.e. all but the last
    return ids.to(DEV), att.to(DEV), pm.to(DEV), cm.to(DEV), dm.to(DEV)


def token_logprobs(items, grad, keep_force=False, want_w=False):
    """Per-token log-probs (at T) of the completion tokens, and per-sequence routed skip fraction over completion tokens.
    keep_force=True leaves the prompt mask set, so a gradient-checkpointing recompute during backward routes exactly as the forward did;
    the caller resets it after backward."""
    ids, att, pm, cm, dm = batch_tensors(items)
    FORCE["mode"] = "train"; FORCE["mask"] = pm
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx, torch.autocast("cuda", dtype=torch.bfloat16):
        o = model.model(input_ids=ids, attention_mask=att)
        tgt = cm[:, 1:]                                                    # position t predicts token t+1
        h = o.last_hidden_state[:, :-1][tgt]                               # [N_completion_tokens, H]
        logits = model.lm_head(h).float() / a.temperature
        lp = torch.log_softmax(logits, -1).gather(-1, ids[:, 1:][tgt].unsqueeze(-1)).squeeze(-1)
    if not keep_force:
        FORCE["mode"] = "native"; FORCE["mask"] = None
    m = torch.stack([x.float() for x in o.router_masks if x is not None], dim=-1)   # [B, T, routed]
    skip = [float(1 - m[r][dm[r]].mean()) if dm[r].any() else 0.0 for r in range(len(items))]
    counts = tgt.sum(1).tolist()
    if want_w:                                                             # router weights sigmoid(logit) at the same decode positions, [n-1, routed]
        w = torch.stack([x.float() for x in o.router_weights if x is not None], dim=-1)
        return lp, counts, skip, [w[r][dm[r]] for r in range(len(items))]
    return lp, counts, skip


def swap_in(src):
    keep = {k: params[k].detach().clone() for k in TRAIN_KEYS}
    with torch.no_grad():
        for k in TRAIN_KEYS:
            params[k].copy_(src[k].float())
    return keep


def swap_back(keep):
    with torch.no_grad():
        for k in TRAIN_KEYS:
            params[k].copy_(keep[k])


# ---- optimiser and state ----
opt = torch.optim.AdamW([params[n] for n in TRAIN_KEYS], lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)
start_it, gstep, history = 0, 0, []
if a.resume:
    ck = torch.load(os.path.join(a.resume, "state.pt"), map_location="cpu", weights_only=False)
    allowed = {x for x in a.allow_cfg_change.split(",") if x}
    diff = {k: (ck["cfg"].get(k), CFG.get(k)) for k in set(ck["cfg"]) | set(CFG) if ck["cfg"].get(k) != CFG.get(k)}
    assert set(diff) <= allowed, ("config differs from the checkpoint's", diff)
    assert ck["ref_id"] == REF_ID, "reference identity differs"
    with torch.no_grad():
        for k in TRAIN_KEYS:
            params[k].copy_(ck["trainable"][k].to(DEV))
    opt.load_state_dict(ck["opt"])
    for g in opt.param_groups:                                             # an allowed lr change takes effect (the state carries the old lr)
        g["lr"] = a.lr
    if "frozen_fp" in ck:                                                  # the frozen backbone is the one this run started from
        assert ck["frozen_fp"] == FROZEN_FP, "frozen tensors differ from the checkpoint's run"
    start_it, gstep, history = ck["iteration"], ck["global_step"], ck["history"]
    if diff:
        history.append({"config_change_at_iteration": start_it, "changes": diff})
        log("CONFIG CHANGE (allowed by --allow-cfg-change)", json.dumps(diff))
    rng = torch.load(os.path.join(a.resume, f"rng_rank{RANK}.pt"), weights_only=False)
    random.setstate(rng["python"]); torch.set_rng_state(rng["torch"]); torch.cuda.set_rng_state(rng["cuda"])
    log(f"resumed {a.resume} at iteration {start_it}, step {gstep}")


SAVED = set()


def save_ckpt(it):
    if it in SAVED:                                                        # e.g. a budget stop right after a periodic checkpoint
        log(f"checkpoint for iteration {it} already written by this run"); return
    name = [None]
    if MAIN:
        d = os.path.join(a.out, f"ckpt_it{it:03d}")
        if os.path.exists(d):                                              # an earlier run's checkpoint is never replaced
            d += time.strftime(".r%Y%m%dT%H%M%SZ", time.gmtime())
        name[0] = d
    if WORLD > 1:
        dist.broadcast_object_list(name, src=0)
    d = name[0]; tmp = d + ".tmp"
    if MAIN:
        if os.path.exists(tmp):                                            # a save cut by an earlier job's limit: kept, renamed
            os.rename(tmp, tmp + time.strftime(".void%Y%m%dT%H%M%SZ", time.gmtime()))
        os.makedirs(tmp, exist_ok=False)
    if WORLD > 1:
        dist.barrier()
    torch.save({"python": random.getstate(), "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state()}, os.path.join(tmp, f"rng_rank{RANK}.pt"))
    if MAIN:
        torch.save({"trainable": {k: params[k].detach().cpu() for k in TRAIN_KEYS}, "opt": opt.state_dict(), "iteration": it, "global_step": gstep,
                    "history": history, "cfg": CFG, "cfg_hash": CFG_HASH, "ref_id": REF_ID, "frozen_fp": FROZEN_FP}, os.path.join(tmp, "state.pt"))
        torch.save({k: params[k].detach().to("cpu", torch.bfloat16) for k in TRAIN_KEYS}, os.path.join(tmp, "router_weights_vskipper_format.pt"))
    if WORLD > 1:
        dist.barrier()
    if MAIN:
        os.rename(tmp, d)
        log(f"checkpoint {d}")
    SAVED.add(it)


def any_rank(flag):
    if WORLD == 1:
        return flag
    t = torch.tensor([1 if flag else 0], device=DEV)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return bool(t.item())


def gather(obj):
    if WORLD == 1:
        return [obj]
    out = [None] * WORLD
    dist.all_gather_object(out, obj)
    return out


def group_norms():
    res = {}
    for g, keys in (("router", ROUTER_KEYS), ("projector", PROJ_KEYS)):
        sq = sum(float(params[k].grad.float().pow(2).sum()) for k in keys if params[k].grad is not None)
        res[g] = math.sqrt(sq) if math.isfinite(sq) else float("inf")
    return res


def dev_eval(it):
    mine = DEV_IDX[:a.dev_limit][RANK::WORLD]
    rec = []
    t = time.time()
    for s in range(0, len(mine), a.gen_micro):
        chunk = mine[s:s + a.gen_micro]
        gens = generate([prompt_ids(i, 1) for i in chunk], sample=False, seed=0)
        for k in range(0, len(gens), a.train_micro):
            part = gens[k:k + a.train_micro]
            _, _, skip = token_logprobs(part, grad=False)
            for i, g, sk in zip(chunk[k:k + a.train_micro], part, skip):
                text = tok.decode(g["completion"], skip_special_tokens=True)
                st, fl = score(text, i)
                rec.append({"dev_index": i, "strict": st, "flexible": fl, "loop": is_loop(text), "cap_hit": not g["ended"],
                            "tokens": len(g["completion"]), "skip_fraction": round(sk, 4)})
    allr = [x for part in gather(rec) for x in part]
    if MAIN:
        with open(os.path.join(a.out, f"dev_it{it:03d}.jsonl"), "w") as f:
            for x in sorted(allr, key=lambda z: z["dev_index"]):
                f.write(json.dumps(x) + "\n")
        n = len(allr)
        summ = {"iteration": it, "n": n, "seconds": round(time.time() - t, 1)} | {k: round(sum(x[k] for x in allr) / n, 4) for k in ("strict", "flexible", "loop", "cap_hit", "tokens", "skip_fraction")}
        summ["layers_used"] = round(NLAYER - len(routed) * summ["skip_fraction"], 3)
        log("DEV", json.dumps(summ))
        with open(os.path.join(a.out, "dev_summary.jsonl"), "a") as f:
            f.write(json.dumps(summ) + "\n")


FOCUS = json.load(open(a.focus_indices)) if a.focus_indices else []
assert bool(FOCUS) == (a.focus_frac > 0), "--focus-indices and --focus-frac go together"
assert len(FOCUS) == len(set(FOCUS)), "focus prompts must be distinct (a repeat would double a prompt's weight)"
assert set(FOCUS) <= set(TRAIN_IDX), "focus prompts must come from the training pool"
FOCUS_SET = set(FOCUS); NON_FOCUS = [i for i in TRAIN_IDX if i not in FOCUS_SET]
if FOCUS and a.focus_frac > 0:
    assert len(NON_FOCUS) >= a.prompts - min(len(FOCUS), round(a.focus_frac * a.prompts)), "too few training-pool prompts outside the focus set"
if 0 in EVAL_AT and start_it == 0:
    dev_eval(0)

it_times = []
for it in range(start_it, a.iterations):
    if a.time_budget_min and it_times and any_rank((time.time() - T0) / 60 + 1.3 * max(it_times) / 60 > a.time_budget_min):
        save_ckpt(it)
        log(f"PAUSED before iteration {it + 1}: time budget"); break
    t_it = time.time()
    rng_it = random.Random(a.seed + it)
    if FOCUS and a.focus_frac > 0:                                             # the same draw on every rank (same seed)
        k = min(len(FOCUS), round(a.focus_frac * a.prompts))
        order = rng_it.sample(FOCUS, k) + rng_it.sample(NON_FOCUS, a.prompts - k)
        rng_it.shuffle(order)
    else:
        order = rng_it.sample(TRAIN_IDX, a.prompts)
    mine = order[RANK::WORLD]
    # 1. rollouts: G samples per prompt from one sampling policy
    torch.cuda.reset_peak_memory_stats(); t = time.time()
    items = []
    flat = [(i, g) for i in mine for g in range(a.group)]
    for s in range(0, len(flat), a.gen_micro):
        chunk = flat[s:s + a.gen_micro]
        gens = generate([prompt_ids(i, 0) for i, _ in chunk], sample=True, seed=a.seed * 1000 + it * 100 + RANK * 10 + s)
        for (i, g), x in zip(chunk, gens):
            items.append(x | {"q": i, "g": g})
    roll_s = time.time() - t; roll_peak = torch.cuda.max_memory_allocated() / 2**30; roll_res = torch.cuda.max_memory_reserved() / 2**30
    # 2. old and reference log-probs (same T, same routing rule), skip fraction, rewards
    torch.cuda.reset_peak_memory_stats(); t = time.time()
    for s in range(0, len(items), a.train_micro):
        part = items[s:s + a.train_micro]
        lp, counts, skip = token_logprobs(part, grad=False)
        keep = swap_in(REF)
        rlp, _, rskip, rw = token_logprobs(part, grad=False, want_w=True)
        swap_back(keep); del keep
        o = 0
        for x, c, sk, rsk, w in zip(part, counts, skip, rskip, rw):
            x["old"], x["ref"], x["skip"], x["skip_ref"] = lp[o:o + c].detach(), rlp[o:o + c].detach(), sk, rsk
            if a.lambda_route:
                x["wref"] = w.detach()
            o += c
    for x in items:
        text = tok.decode(x["completion"], skip_special_tokens=True)
        st, fl = score(text, x["q"]); lo = is_loop(text); cap = not x["ended"]; n = len(x["completion"])
        x.update(strict=st, flexible=fl, loop=lo, cap=cap, n=n)
        x["terms"] = {"correct": a.w_correct * (st or fl), "loop": -a.w_loop * lo, "cap": -a.w_cap * cap, "length": -a.w_len * max(0, n - a.len_ref),
                      "skip": a.w_skip * x["skip"], "skip_ret": -a.w_skip_ret * max(0.0, x["skip_ref"] - x["skip"])}
        x["r"] = sum(x["terms"].values())
    for i in mine:                                                             # group-relative length term (each rank holds whole groups)
        grp = [x for x in items if x["q"] == i]
        mn = sum(x["n"] for x in grp) / len(grp)
        for x in grp:
            x["terms"]["length_group"] = -a.w_len_group * (x["n"] / max(mn, 1.0) - 1.0)
            x["r"] += x["terms"]["length_group"]
    for i in mine:
        grp = [x for x in items if x["q"] == i]
        rs = torch.tensor([x["r"] for x in grp]); mu, sd = rs.mean(), rs.std(unbiased=False)
        for x in grp:
            x["adv"] = float((x["r"] - mu) / (sd + 1e-4))
    os.makedirs(os.path.join(a.out, "rollouts"), exist_ok=True)
    with open(os.path.join(a.out, "rollouts", f"it{it + 1:03d}_rank{RANK}.jsonl"), "w") as f:
        for x in items:
            f.write(json.dumps({k: x[k] for k in ("q", "g", "n", "ended", "strict", "flexible", "loop", "cap", "skip", "skip_ref", "r", "adv", "terms", "completion")}) + "\n")
    lp_s = time.time() - t; lp_peak = torch.cuda.max_memory_allocated() / 2**30; lp_res = torch.cuda.max_memory_reserved() / 2**30
    # 3. update: mini-batches, gradients all-reduced, router and projector gradients checked separately
    model.train(); torch.cuda.reset_peak_memory_stats(); t = time.time()
    perm = random.Random(a.seed * 31 + it).sample(range(len(items)), len(items))
    mb = math.ceil(len(items) / a.mini_batches); stats = {"kl": 0.0, "clipfrac": 0.0, "tok": 0.0, "hinge": 0.0, "up": 0.0, "dec": 0.0}; gn_log = []
    for m in range(a.mini_batches):
        sel = [items[j] for j in perm[m * mb:(m + 1) * mb]]
        ntok = float(sum(x["n"] for x in sel))
        if WORLD > 1:                                                      # the loss is a mean over the mini-batch's tokens on ALL ranks
            nt = torch.tensor([ntok], device=DEV, dtype=torch.float64); dist.all_reduce(nt); ntok = float(nt.item())
        opt.zero_grad(set_to_none=True)
        for s in range(0, len(sel), a.train_micro):
            part = sel[s:s + a.train_micro]
            if a.lambda_route:
                lp, counts, _, ws = token_logprobs(part, grad=True, keep_force=True, want_w=True)
            else:
                lp, counts, _ = token_logprobs(part, grad=True, keep_force=True)
            old = torch.cat([x["old"] for x in part]); ref = torch.cat([x["ref"] for x in part])
            adv = torch.cat([torch.full((c,), x["adv"], device=DEV) for x, c in zip(part, counts)])
            ratio = torch.exp(lp - old)
            surr = torch.min(ratio * adv, torch.clamp(ratio, 1 - a.clip, 1 + a.clip) * adv)
            kl = torch.exp(ref - lp) - (ref - lp) - 1
            loss = (-surr + a.beta * kl).sum() / max(ntok, 1)
            if a.lambda_route:                                             # per decision: penalize a move toward RUN relative to the released router
                d = torch.cat([w - x["wref"] for w, x in zip(ws, part)])
                hinge = torch.relu(d)
                loss = loss + a.lambda_route * hinge.sum() / (len(routed) * max(ntok, 1))
                stats["hinge"] += float(hinge.sum()); stats["up"] += float((d > 0).sum()); stats["dec"] += float(d.numel())
            loss.backward()
            FORCE["mode"] = "native"; FORCE["mask"] = None
            stats["kl"] += float(kl.sum()); stats["clipfrac"] += float(((ratio - 1).abs() > a.clip).sum()); stats["tok"] += float(lp.numel())
        if WORLD > 1:
            for k in TRAIN_KEYS:
                if params[k].grad is None:
                    params[k].grad = torch.zeros_like(params[k])
                dist.all_reduce(params[k].grad)                            # sum: each rank's loss is already divided by the global count
        gr = group_norms()
        if not all(0 < gr[g] < float("inf") for g in ("router", "projector")):
            raise FloatingPointError(f"iteration {it + 1} mini-batch {m}: router/projector gradient norms {gr}")
        gn = torch.nn.utils.clip_grad_norm_([params[k] for k in TRAIN_KEYS], 1.0)
        if not math.isfinite(float(gn)):
            raise FloatingPointError(f"nonfinite gradient norm {float(gn)}")
        opt.step(); gstep += 1; gn_log.append({"router": round(gr["router"], 6), "projector": round(gr["projector"], 6), "total": round(float(gn), 6)})
    upd_s = time.time() - t; upd_peak = torch.cuda.max_memory_allocated() / 2**30; upd_res = torch.cuda.max_memory_reserved() / 2**30
    # 4. log (gathered over ranks)
    part = {"n": len(items), "strict": sum(x["strict"] for x in items), "correct": sum(x["strict"] or x["flexible"] for x in items),
            "loop": sum(x["loop"] for x in items), "cap": sum(x["cap"] for x in items), "tokens": sum(x["n"] for x in items),
            "skip": sum(x["skip"] for x in items), "r": sum(x["r"] for x in items), "kl": stats["kl"], "clip": stats["clipfrac"], "tok": stats["tok"],
            "peaks": [roll_peak, lp_peak, upd_peak], "reserved": [roll_res, lp_res, upd_res],
            "terms": {k: sum(x["terms"][k] for x in items) for k in ("correct", "loop", "cap", "length", "skip", "length_group", "skip_ret")},
            "skip_ref": sum(x["skip_ref"] for x in items), "hinge": stats["hinge"], "up": stats["up"], "dec": stats["dec"],
            "flat_groups": sum(1 for i in mine if len({x["r"] for x in items if x["q"] == i}) == 1), "groups": len(mine)}
    g = gather(part)
    if MAIN:
        tot = {k: sum(p[k] for p in g) for k in ("n", "strict", "correct", "loop", "cap", "tokens", "skip", "r", "kl", "clip", "tok", "skip_ref", "hinge", "up", "dec")}
        rec = {"iteration": it + 1, "global_step": gstep, "rollouts": tot["n"], "correct_rate": round(tot["correct"] / tot["n"], 4),
               "strict_rate": round(tot["strict"] / tot["n"], 4), "loop_rate": round(tot["loop"] / tot["n"], 4), "cap_rate": round(tot["cap"] / tot["n"], 4),
               "mean_tokens": round(tot["tokens"] / tot["n"], 1), "skip_fraction": round(tot["skip"] / tot["n"], 4),
               "skip_fraction_released_router": round(tot["skip_ref"] / tot["n"], 4),
               "route_hinge_mean": round(tot["hinge"] / max(tot["dec"], 1), 6), "route_up_share": round(tot["up"] / max(tot["dec"], 1), 4),
               "layers_used": round(NLAYER - len(routed) * tot["skip"] / tot["n"], 3), "mean_reward": round(tot["r"] / tot["n"], 4),
               "mean_kl": round(tot["kl"] / max(tot["tok"], 1), 6), "clip_fraction": round(tot["clip"] / max(tot["tok"], 1), 4), "grad_norms": gn_log,
               "seconds": {"rollout": round(roll_s, 1), "logprob_ref_reward": round(lp_s, 1), "update": round(upd_s, 1), "iteration": round(time.time() - t_it, 1)},
               "peak_gib_max_over_ranks": {"rollout": round(max(p["peaks"][0] for p in g), 2), "logprob": round(max(p["peaks"][1] for p in g), 2),
                                           "update": round(max(p["peaks"][2] for p in g), 2)},
               "peak_reserved_gib_max_over_ranks": {"rollout": round(max(p["reserved"][0] for p in g), 2),
                                                    "logprob": round(max(p["reserved"][1] for p in g), 2), "update": round(max(p["reserved"][2] for p in g), 2)},
               "reward_terms_mean": {k: round(sum(p["terms"][k] for p in g) / tot["n"], 4) for k in ("correct", "loop", "cap", "length", "skip", "length_group", "skip_ret")},
               "groups_with_zero_reward_spread": f'{sum(p["flat_groups"] for p in g)}/{sum(p["groups"] for p in g)}'}
        history.append(rec)
        log("ITER", json.dumps(rec))
        with open(os.path.join(a.out, "iterations.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")
    it_times.append(time.time() - t_it)
    if (it + 1) % a.ckpt_every == 0 or it + 1 == a.iterations:
        changed = [n for n in FROZEN if fingerprint(params[n]) != FROZEN_FP[n]]
        if changed:
            raise RuntimeError(f"frozen tensors changed: {changed[:5]}")
        save_ckpt(it + 1)
    if (it + 1) in EVAL_AT:
        dev_eval(it + 1)
log("DONE", f"elapsed {round((time.time() - T0) / 60, 1)} min")
if WORLD > 1:
    dist.destroy_process_group()
