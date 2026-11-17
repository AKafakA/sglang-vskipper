#!/usr/bin/env bash
# cloudlab_skipret_run.sh ARM [ITERATIONS] [RUN_TAG] -- the skip-retention GRPO branch (a branch from the released
# checkpoint, a penalty for losing skip relative to released, repetition and length objectives kept) on the CloudLab node, after
# cloudlab_fdfix_setup.sh system / env / inputs / verify. One 4-GPU torchrun; the released router/projector is the start (no --resume on the
# first launch; later launches resume the run's newest checkpoint). The base reward is the length arm's:
#   r = 1 correct - 1 loop - 0.5 cap - 0.001 x max(0, n - 256) + 0.2 skip - 0.5 x (n / group mean n - 1), focus 0.5 on the 471 loop-prone
#   training prompts, 32 prompts x 8 at T 0.8, max_new 1,024, lr 5e-6, beta 0.04, clip 0.2, 2 optimizer steps per iteration.
# ARM adds the skip retention (the owner picks the arm):
#   B1 : --lambda-route 1.0    router loss: relu(router weight - released router weight) per decision, mean over decode x routed layers
#   B4 : --lambda-route 4.0
#   A2 : --w-skip-ret 2.0      reward: -2 x max(0, released router's executed skip on the sample - the sample's executed skip)
#   A2B1 : both
# Training pool only (pool.json train_indices); the test split is not on the node. Checkpoints every 3 iterations under /mydata/fdfix/runs.
# RUN_TAG (third argument, default 20261115) names the run directory runs/skipret-<ARM>-<RUN_TAG>: a smoke run uses its own tag so the
# real run never resumes from it (the trainer always writes a checkpoint at its last iteration).
# DRY_RUN=1 prints the command. usage (on the node): bash cloudlab_skipret_run.sh B1 48   |   smoke: bash cloudlab_skipret_run.sh A2B1 2 smoke40g
set -euo pipefail
ARM=${1:?B1|B4|A2|A2B1}; IT=${2:-48}; TAG=${3-20261115}; DRY=${DRY_RUN:-}
[[ $TAG =~ ^[A-Za-z0-9]+$ ]] || { echo "bad run tag $TAG"; exit 2; }
O=/mydata/fdfix; V=$O/venv/bin; M=$O/models/FlexiDepth-Llama-3-8B-Instruct; T=$O/tools; DA=$O/data
case $ARM in
  B1) EXTRA=(--lambda-route 1.0);; B4) EXTRA=(--lambda-route 4.0);; A2) EXTRA=(--w-skip-ret 2.0);; A2B1) EXTRA=(--w-skip-ret 2.0 --lambda-route 1.0);;
  *) echo "unknown arm $ARM"; exit 2;;
esac
RUN=$O/runs/skipret-$ARM-$TAG; mkdir -p "$RUN"
for f in "$M/config.json" "$O/serving/flexidepth_router_weights.pt" "$DA/pool.json" "$DA/focus_loops_released_greedy.json" "$DA/gsm8k-rows/train.jsonl" "$T/fdfix_grpo.py"; do
  [ -f "$f" ] || { echo "missing $f"; exit 2; }; done
[ ! -e "$DA/gsm8k-rows/test.jsonl" ] || { echo "a test split is present on the node: refusing"; exit 3; }
R=$(ls -d "$RUN"/ckpt_it[0-9][0-9][0-9] 2>/dev/null | sort | tail -1 || true)
RES=(); [ -n "$R" ] && RES=(--resume "$R")
CMD=("$V/torchrun" --standalone --nproc_per_node 4 "$T/fdfix_grpo.py" --model "$M" --rows "$DA/gsm8k-rows" --pool "$DA/pool.json"
     --released-pt "$O/serving/flexidepth_router_weights.pt" --prompts 32 --group 8 --temperature 0.8 --max-new 1024 --gen-micro 32
     --train-micro 2 --mini-batches 2 --lr 5e-6 --beta 0.04 --clip 0.2 --focus-indices "$DA/focus_loops_released_greedy.json" --focus-frac 0.5
     --w-len-group 0.5 "${EXTRA[@]}" --allow-cfg-change "" --out "$RUN" "${RES[@]}" --iterations "$IT" --ckpt-every 3 --eval-at "" --dev-limit 500)
if [ -n "$DRY" ]; then printf '%q ' "${CMD[@]}"; echo; exit 0; fi
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HOME="$RUN/hf_home" HF_MODULES_CACHE="$RUN/hf_modules"
nvidia-smi --query-gpu=index,name,memory.total,power.limit --format=csv > "$RUN/gpus_$(date -u +%Y%m%dT%H%M%SZ).csv"
nohup "${CMD[@]}" >> "$RUN/train_$(date -u +%Y%m%dT%H%M%SZ).log" 2>&1 < /dev/null &
echo "started pid $! arm $ARM to iteration $IT in $RUN"
