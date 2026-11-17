"""fdfix_lib.py -- shared pieces for the GRPO trainer and the held-out evaluator (FlexiDepth-Llama-3-8B-Instruct).

Prompt construction, scoring and the line-repeat loop detector are copied verbatim from fdfix_grpo.py, so the trainer and the evaluator
see identical prompts (training prompts: salt 0; dev prompts: salt 1; demonstrations only from the training pool).
find_inner_repetition / _verify_repetition_at are ported from Liquid4All/antidoom src/antidoom/repetition.py (Apache-2.0).
Routing control: dense prompt, routed decode. Router logit 30 (sigmoid -> 1.0, mask 1, weight 1) is exactly the base layer.
"""
import hashlib
import json
import os
import random
import re

import torch

SEED = 20261114


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


# ---------------- distributed helpers ----------------
class Dist:
    def __init__(self):
        import torch.distributed as dist
        self.dist = dist
        self.world = int(os.environ.get("WORLD_SIZE", "1")); self.rank = int(os.environ.get("RANK", "0"))
        self.local = int(os.environ.get("LOCAL_RANK", "0"))
        if self.world > 1:                                                      # ranks finish generation minutes apart
            from datetime import timedelta
            dist.init_process_group("nccl", timeout=timedelta(minutes=90))
        torch.cuda.set_device(self.local)
        self.dev = torch.device("cuda", self.local)
        self.main = self.rank == 0

    def barrier(self):
        if self.world > 1:
            self.dist.barrier()

    def gather(self, obj):
        if self.world == 1:
            return [obj]
        out = [None] * self.world
        self.dist.all_gather_object(out, obj)
        return out

    def close(self):
        if self.world > 1:
            self.dist.destroy_process_group()


# ---------------- model and routing control ----------------
AMP = torch.bfloat16          # compute dtype of every forward here (weights and autocast); fdfix_eval.py --dtype fp16 sets float16


def load_model(model_dir, local):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True, dtype=AMP, use_cache=False,
                                                 device_map={"": local})
    return tok, model


class Routing:
    """mode 'gen': a multi-token forward (the generation prefill) forces dense the positions in `mask` (all positions when mask is None);
    single-token decode steps route natively. mode 'train': positions in `mask` forced dense, the rest routed. mode 'native': untouched."""

    def __init__(self, model):
        self.mode, self.mask, self.capture = "native", None, None
        self.decode_dense = False                                              # generation only: decode steps forced dense (= the base layer)
        self.layers = list(model.config.routing_layers); self.n_layers = model.config.num_hidden_layers
        for i in self.layers:
            self._wrap(model.model.layers[i].router)

    def _wrap(self, mod):
        orig = mod.forward

        def fwd(h):
            logits = orig(h)
            if self.mode == "gen" and h.shape[1] == 1 and self.decode_dense:       # the served stock decode: every layer runs
                logits = torch.full_like(logits, 30.0)
            if self.mode == "gen" and h.shape[1] == 1 and self.capture is not None:   # a decode step: the layer computes exactly this
                self.capture.append((torch.sigmoid(logits) > 0.5).reshape(-1))
            if self.mode == "gen" and h.shape[1] > 1:
                if self.mask is None:
                    return torch.full_like(logits, 30.0)
                return torch.where(self.mask[..., None], torch.full_like(logits, 30.0), logits)
            if self.mode == "train" and self.mask is not None:
                return torch.where(self.mask[..., None], torch.full_like(logits, 30.0), logits)
            return logits
        mod.forward = fwd

    def set(self, mode, mask=None):
        self.mode, self.mask = mode, mask

    def reset(self):
        self.mode, self.mask = "native", None


# ---------------- data, prompts, scoring (verbatim from fdfix_grpo.py) ----------------
def load_rows(rows_dir, pool_path, need_dev, dev_limit):
    pool = json.load(open(pool_path))
    train_idx, dev_idx = pool["train_indices"], pool["dev_indices"]
    assert not set(train_idx) & set(dev_idx)
    need = set(train_idx) | (set(dev_idx[:dev_limit]) if need_dev else set())
    rows = {}
    with open(os.path.join(rows_dir, "train.jsonl")) as f:
        for i, line in enumerate(f):
            if i in need:
                rows[i] = json.loads(line)
    assert len(rows) == len(need), (len(rows), len(need))
    return rows, train_idx, dev_idx


def norm_ans(s):
    return re.sub(r"\.$", "", re.sub(r"[,$]", "", s.strip()))


class Prompts:
    def __init__(self, tok, rows, train_idx, seed=SEED):
        self.tok, self.rows, self.train_idx, self.seed = tok, rows, train_idx, seed

    def messages(self, i, rng):
        demos = rng.sample([j for j in self.train_idx if j != i], 5)          # demonstrations only from the training pool
        m = []
        for j in demos:
            m += [{"role": "user", "content": f"Question: {self.rows[j]['question'].strip()}\nAnswer:"},
                  {"role": "assistant", "content": self.rows[j]["answer"]}]
        return m + [{"role": "user", "content": f"Question: {self.rows[i]['question'].strip()}\nAnswer:"}]

    def ids(self, i, salt):
        return self.tok.apply_chat_template(self.messages(i, random.Random(self.seed * 7919 + i * 31 + salt)), add_generation_prompt=True,
                                            tokenize=True)

    def gold(self, i):
        return norm_ans(self.rows[i]["answer"].split("####")[-1])


STRICT = re.compile(r"#### (\-?[0-9\.\,]+)"); FLEX = re.compile(r"(-?[$0-9.,]{2,})|(-?[0-9]+)")


def score(text, gold):
    s = STRICT.search(text); f = FLEX.findall(text)
    strict = s is not None and norm_ans(s.group(1)) == gold
    flex = bool(f) and norm_ans([x for x in f[-1] if x][0]) == gold
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


# ---------------- Antidoom's in-output repetition detector (ported, Apache-2.0) ----------------
REP_CFG = dict(min_repeats=4, max_period=1024, min_period=1, min_total_repeated=60, sample_len=16, sample_interval=128)


def _verify_repetition_at(text, start_pos, period, min_repeats, min_total_repeated):
    if period < 1 or start_pos < 0 or start_pos + period > len(text):
        return None
    pattern = text[start_pos:start_pos + period]
    reps, pos = 0, start_pos
    while pos + period <= len(text) and text[pos:pos + period] == pattern:
        reps += 1; pos += period
    end_pos = pos
    pos = start_pos - period
    while pos >= 0 and text[pos:pos + period] == pattern:
        reps += 1; start_pos = pos; pos -= period
    if reps >= min_repeats and reps * period >= min_total_repeated:
        return {"start": start_pos, "end": end_pos, "period": period, "repeats": reps, "snippet": pattern[:100]}
    return None


def find_inner_repetition(text, min_repeats=4, max_period=1024, min_period=1, min_total_repeated=60, sample_len=16, sample_interval=128):
    if not text or len(text) < min_total_repeated:
        return None
    n = len(text)
    for sp in range(0, n - sample_len, sample_interval):
        fp = text[sp:sp + sample_len]
        op = text.find(fp, sp + sample_len)
        if op != -1 and min_period <= op - sp <= max_period:
            hit = _verify_repetition_at(text, sp, op - sp, min_repeats, min_total_repeated)
            if hit:
                return hit
        op = text.rfind(fp, 0, sp)
        if op != -1 and min_period <= sp - op <= max_period:
            hit = _verify_repetition_at(text, op, sp - op, min_repeats, min_total_repeated)
            if hit:
                return hit
    return None


# ---------------- generation ----------------
@torch.no_grad()
def generate(model, R, tok, prompts, max_new, dev, sample=False, temperature=1.0, seed=0, dense_lens=None, loop_stop=False,
             prompt_routed=False, decode_dense=False):
    """Left-padded HF generate. dense_lens[r] = number of leading tokens of prompts[r] forced dense in the prefill (default: all, i.e. a
    dense prompt); the remaining prefill positions and every decode step route natively. loop_stop: a row whose output already shows
    Antidoom's repetition (>= 4 repeats, >= 60 characters) stops early, so the pairs stage finds the same repetition -- mining only.
    Served regimes (round 3): prompt_routed routes every prompt position natively, as the served FD prefill does (padding stays dense);
    decode_dense forces every decode step dense, which is the base model's layer, as the served stock decode."""
    assert not (prompt_routed and decode_dense), "one served regime per run"   # Codex review: the combination is not defined here
    if prompt_routed:
        assert dense_lens is None, "prompt_routed replaces dense_lens"
        dense_lens = [0] * len(prompts)
    from transformers import StoppingCriteria, StoppingCriteriaList
    eot = tok.convert_tokens_to_ids("<|eot_id|>")
    L = max(map(len, prompts))
    ids = torch.full((len(prompts), L), eot, dtype=torch.long); att = torch.zeros_like(ids)
    mask = None
    if dense_lens is not None:
        mask = torch.zeros((len(prompts), L), dtype=torch.bool)
    for r, p in enumerate(prompts):
        ids[r, L - len(p):] = torch.tensor(p); att[r, L - len(p):] = 1
        if mask is not None:
            mask[r, L - len(p):L - len(p) + dense_lens[r]] = True
            mask[r, :L - len(p)] = True                                        # padding: irrelevant, kept dense
    stop_at = [None] * len(prompts)

    class LoopStop(StoppingCriteria):
        def __call__(self, input_ids, scores, **kw):
            done = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
            n = input_ids.shape[1] - L
            if n == 0 or n % 32:
                return done | torch.tensor([s is not None for s in stop_at], device=input_ids.device)
            for r, row in enumerate(input_ids[:, L:].tolist()):
                if stop_at[r] is not None:
                    done[r] = True; continue
                if eot in row:
                    continue
                text = tok.decode(row, skip_special_tokens=True)
                if find_inner_repetition(text, **REP_CFG):
                    stop_at[r] = n; done[r] = True
            return done

    model.eval(); R.set("gen", mask.to(dev) if mask is not None else None); R.capture = []
    kw = dict(do_sample=True, temperature=temperature, top_p=1.0, top_k=0) if sample else dict(do_sample=False)
    try:
        R.decode_dense = decode_dense; torch.manual_seed(seed)                # inside the try: always cleared in the finally
        with torch.autocast("cuda", dtype=AMP):
            out = model.generate(input_ids=ids.to(dev), attention_mask=att.to(dev), max_new_tokens=max_new, use_cache=True, pad_token_id=eot,
                                 eos_token_id=eot, stopping_criteria=StoppingCriteriaList([LoopStop()]) if loop_stop else None, **kw)
    finally:
        cap, R.capture = R.capture, None
        R.reset(); R.decode_dense = False
    nr = len(R.layers)
    assert len(cap) % nr == 0, (len(cap), nr)
    dec = torch.stack(cap).view(len(cap) // nr, nr, len(prompts)).float() if cap else None   # [decode steps, routed layers, rows]
    res = []
    for r, p in enumerate(prompts):
        new = out[r, L:].tolist()
        if stop_at[r] is not None:
            g = {"completion": new[:stop_at[r]], "ended": False, "loop_stopped": True}
        else:
            ended = eot in new
            g = {"completion": new[:new.index(eot) + 1] if ended else new, "ended": ended, "loop_stopped": False}
        k = len(g["completion"]) - 1                                       # decode step j feeds completion token j-1; the last is never fed
        g["exec_decisions"] = k * nr
        g["exec_skip"] = float(1 - dec[:k, :, r].mean()) if k > 0 else 0.0
        res.append(g)
    return res


def teacher_forced(model, R, seqs, dense_lens, dev, grad=False):
    """Right-padded teacher-forced forward of full sequences with the first dense_lens[r] positions forced dense and the rest routed.
    Returns (last-position logits fp32 [B, V], hidden states, router masks [B, T, n_routed], lengths). The routing mask stays set until
    the caller resets R (so a backward recompute, if any, routes as the forward did)."""
    L = max(map(len, seqs))
    ids = torch.zeros((len(seqs), L), dtype=torch.long); att = torch.zeros_like(ids); pm = torch.zeros((len(seqs), L), dtype=torch.bool)
    for r, s in enumerate(seqs):
        ids[r, :len(s)] = torch.tensor(s); att[r, :len(s)] = 1; pm[r, :dense_lens[r]] = True
    ids, att, pm = ids.to(dev), att.to(dev), pm.to(dev)
    R.set("train", pm)
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx, torch.autocast("cuda", dtype=AMP):
        o = model.model(input_ids=ids, attention_mask=att)
        last = torch.tensor([len(s) - 1 for s in seqs], device=dev)
        h = o.last_hidden_state[torch.arange(len(seqs), device=dev), last]
        logits = model.lm_head(h).float()
    masks = torch.stack([x.float() for x in o.router_masks if x is not None], dim=-1)
    return logits, masks, [len(s) for s in seqs]


def model_identity(d):
    h = hashlib.sha256()
    names = ["config.json", "generation_config.json", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json",
             "special_tokens_map.json"] + sorted(f for f in os.listdir(d) if f.endswith(".py"))
    for n in names:
        p = os.path.join(d, n)
        h.update(n.encode() + (sha_file(p).encode() if os.path.exists(p) else b"absent"))
    return h.hexdigest()[:16]
