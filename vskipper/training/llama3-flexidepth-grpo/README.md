# GRPO post-training of the FlexiDepth Llama-3-8B router and projector

This directory holds the trainer and the node-side scripts that produced the post-trained FlexiDepth Llama-3-8B checkpoint of the
paper's natural-stopping study (108 iterations). GRPO updates only the router and projector (209,784,832 parameters). The
backbone is frozen. The serving code is unchanged: the checkpoint is served by the arm `vskipper_cand_grpo`, which is the served
design with another router/projector file.

## Files

| file | role |
|---|---|
| `fdfix_grpo.py` | the trainer, as run |
| `fdfix_lib.py` | model loading, prompts, scoring and repetition detectors used by the trainer and the evaluator |
| `fdfix_eval.py` | greedy evaluation on the 500 held-out training-split rows |
| `run/cloudlab_fdfix_setup.sh` | node bring-up: system, environment from `run/fdfix-freeze.txt`, inputs, verification |
| `run/cloudlab_skipret_run.sh` | the training launcher (arm `B4` is the reported run) |
| `run/b4_segments2.sh` | runs the segments to iterations 60, 72, 84 and 96 with a held-out check after each |
| `run/cloudlab_dev_eval.sh` | the held-out check |
| `data/pool.json` | training indices (4,691) and held-out indices (500) of the GSM8K training split |
| `data/focus_loops_released_greedy.json` | the 471 training prompts on which the released checkpoint repeats |
| `data/shard_oids_2ce73595.json` | sha256 of the released checkpoint shards, checked by the setup script |
| `data/gsm8k_split_check.py`, `data/report.json`, `data/source.json`, `data/gsm8k_rows.sha256` | split identities, overlap checks and the pool builder |
| `iterations.jsonl` | the per-iteration training record of the reported run (108 lines) |
| `SHA256SUMS` | hashes of the files above |

## Inputs

- The released FlexiDepth Llama-3-8B checkpoint directory and its router/projector file
  (`flexidepth_router_weights.pt`, sha256 `9a1759073d4b68ad24f249f5c303a7d74c9b51b0289a6e96812dcb5d3ffed6b9`).
- GSM8K `main` rows as JSONL (`train.jsonl`, sha256 in `data/gsm8k_rows.sha256`). The test split is never placed on the training node,
  and the launcher refuses to start if it is present.
- Four A100-SXM4-40GB GPUs.

## Commands, as run

```bash
# stage data/, the three .py files (as tools/), the released router file (as serving/), the freeze file and a MANIFEST.sha256
# of them under /mydata/fdfix/_incoming, then:
bash run/cloudlab_fdfix_setup.sh system      # may ask for a reboot, then run it again
bash run/cloudlab_fdfix_setup.sh env
bash run/cloudlab_fdfix_setup.sh inputs
bash run/cloudlab_fdfix_setup.sh verify

bash run/cloudlab_skipret_run.sh B4 48       # iterations 1 to 48
nohup bash run/b4_segments2.sh > b4_segments2.log 2>&1 < /dev/null &   # segments to 60, 72, 84, 96
bash run/cloudlab_skipret_run.sh B4 108      # iterations 97 to 108
bash run/cloudlab_dev_eval.sh ckpt skipret-B4-<run tag> 108
```

`DRY_RUN=1 bash run/cloudlab_skipret_run.sh B4 108` prints the trainer command. For the reported run it is:

```bash
torchrun --standalone --nproc_per_node 4 fdfix_grpo.py --model <FlexiDepth-Llama-3-8B-Instruct> --rows <gsm8k-rows> --pool data/pool.json \
  --released-pt <flexidepth_router_weights.pt> --prompts 32 --group 8 --temperature 0.8 --max-new 1024 --gen-micro 32 \
  --train-micro 2 --mini-batches 2 --lr 5e-6 --beta 0.04 --clip 0.2 --focus-indices data/focus_loops_released_greedy.json --focus-frac 0.5 \
  --w-len-group 0.5 --lambda-route 4.0 --allow-cfg-change "" --out <run-dir> [--resume <run-dir>/ckpt_itNNN] --iterations 108 \
  --ckpt-every 3 --eval-at "" --dev-limit 500
```

Each iteration samples 8 completions for 32 questions and makes two updates (216 in total). The six training segments took 188.1
minutes in total.

## Output

`<run-dir>/ckpt_it108/router_weights_vskipper_format.pt` is the served file: 419,608,953 bytes,
sha256 `fbcbf1eb8d3e19d80acd117902ce7ab51b5b6fefa51626c364e5f405db7ad4fd`. It holds the trained router and projector tensors in the
layout the serving runtime loads.

## Serving the checkpoint

Add the file to the host JSON under the key `flexidepth_weights_cand_grpo` and serve the arm `vskipper_cand_grpo`
(`vskipper/src/vskipper/runtime/design.py`). The natural-stopping cells use `vskipper/src/vskipper/experiments/run_natural_harvest.py` with the suite
`gsm8k.d179` at 8.25, 10.45 and 13.75 requests/s, three repetitions, beside `upstream_g1024` and `vskipper` in the same runs.
`vskipper/src/vskipper/analysis/revision/README.md` lists the analysis commands.
