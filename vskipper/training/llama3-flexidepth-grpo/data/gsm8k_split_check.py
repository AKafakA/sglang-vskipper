#!/usr/bin/env python3
"""gsm8k_split_check.py -- split identities, overlap checks and the training pool for GSM8K post-training (owner 2026-11-14).

Owner: GSM8K post-training only on the training split, never the evaluated test split or its answers. Verify dataset provenance, split
identities and any sample overlap before training, and keep test data out of training, tuning and model selection.

Inputs:
  --rows DIR      JSONL dumps of openai/gsm8k main (train.jsonl, test.jsonl), one {"row_idx", "question", "answer"} per line, from the
                  datasets-server rows API at a recorded dataset sha
  --suite FILE    the evaluation suite's metadata (gsm8k.d179.metadata.jsonl): request ids gsm8k:<split>:<index>, gold answers,
                  few-shot indices
Outputs (--out DIR): pool.json (training indices and the held-out dev indices), report.json (every check and count).
Fails (exit 1) if a suite gold answer disagrees with the dataset row its id names, or if any pool question matches a test question.
"""
import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path


def gold(answer):
    return answer.split("####")[-1].strip().replace(",", "")


def norm(text):
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())


def ngrams(text, n=13):
    w = norm(text).split()
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True); ap.add_argument("--suite", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--dev", type=int, default=500); ap.add_argument("--seed", type=int, default=20261114)
    a = ap.parse_args()
    rows = {s: [json.loads(l) for l in open(Path(a.rows) / f"{s}.jsonl")] for s in ("train", "test")}
    for s, rs in rows.items():
        if [r["row_idx"] for r in rs] != list(range(len(rs))):
            sys.exit(f"{s}: row indices are not 0..{len(rs) - 1}")
    meta = [json.loads(l) for l in open(a.suite) if l.strip()]
    report = {"dataset_rows": {s: len(r) for s, r in rows.items()}, "suite_rows": len(meta)}

    # 1. split identity: every suite id names a dataset row whose gold answer equals the suite's gold
    bad, used = [], {"train": set(), "test": set()}
    demo = set()
    for m in meta:
        _, split, idx = m["request_id"].split(":")
        idx = int(idx)
        used[split].add(idx)
        g = str(m.get("gold") or (m.get("evaluator_data") or {}).get("gold") or "")
        if gold(rows[split][idx]["answer"]) != g:
            bad.append(m["request_id"])
        demo.update((m.get("evaluator_data") or {}).get("fewshot_indices") or [])
    report["suite_by_split"] = {s: len(v) for s, v in used.items()}
    report["gold_mismatches"] = bad[:20]
    report["gold_mismatch_count"] = len(bad)
    report["fewshot_demo_train_indices"] = len(demo)

    # 2. the training pool: train rows that are not evaluated rows of the suite
    pool = sorted(set(range(len(rows["train"]))) - used["train"])
    # 3. overlap of pool questions with the test split: exact normalised text, and any shared 13-gram
    test_norm = {norm(r["question"]): r["row_idx"] for r in rows["test"]}
    test_grams = {}
    for r in rows["test"]:
        for g in ngrams(r["question"]):
            test_grams.setdefault(g, set()).add(r["row_idx"])
    exact = [i for i in pool if norm(rows["train"][i]["question"]) in test_norm]
    gram_hits = {i: sorted({t for g in ngrams(rows["train"][i]["question"]) for t in test_grams.get(g, ())}) for i in pool}
    gram_hits = {i: t for i, t in gram_hits.items() if t}
    report["pool_size_before_exclusion"] = len(pool)
    report["pool_exact_test_matches"] = exact
    report["pool_13gram_test_overlaps"] = len(gram_hits)
    # rows sharing a 13-gram with any test question are excluded too, so no test wording reaches training
    excluded = set(exact) | set(gram_hits)
    pool = [i for i in pool if i not in excluded]
    # 4. the same check against the suite's evaluated train rows (they are evaluated, so their wording stays out too)
    suite_train_norm = {norm(rows["train"][i]["question"]) for i in used["train"]}
    report["pool_exact_suite_train_matches"] = [i for i in pool if norm(rows["train"][i]["question"]) in suite_train_norm]
    pool = [i for i in pool if norm(rows["train"][i]["question"]) not in suite_train_norm]
    report["pool_size"] = len(pool)
    report["pool_rows_also_used_as_fewshot_demos_in_eval_prompts"] = len(set(pool) & demo)
    # 5. held-out dev set for model selection, from the pool only (the test split is never used for selection)
    rng = random.Random(a.seed)
    dev = sorted(rng.sample(pool, a.dev))
    train = sorted(set(pool) - set(dev))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    pj = {"train_indices": train, "dev_indices": dev, "seed": a.seed, "excluded_for_test_overlap": sorted(excluded),
          "train_sha256": hashlib.sha256(json.dumps(train).encode()).hexdigest(), "dev_sha256": hashlib.sha256(json.dumps(dev).encode()).hexdigest()}
    (out / "pool.json").write_text(json.dumps(pj) + "\n")
    report.update({"train_count": len(train), "dev_count": len(dev), "train_sha256": pj["train_sha256"], "dev_sha256": pj["dev_sha256"]})
    (out / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("pool_exact_test_matches", "gold_mismatches")}, indent=1))
    if bad or exact and set(exact) & set(train + dev):
        sys.exit(1)


if __name__ == "__main__":
    main()
