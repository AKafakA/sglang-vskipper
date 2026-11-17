#!/usr/bin/env bash
# cloudlab_dev_eval.sh baselines | ckpt RUN ITER... -- dev checks on the CloudLab node with the shared evaluator (fdfix_eval.py, the same
# code, 500 dev rows of the training split's pool, greedy, max 1,024 new tokens, bf16), so every CloudLab candidate is paired with
# baselines measured on the same node:
#   baselines : released (dense prompt, routed decode)              -> eval/released-dev500
#               released, routed prompt (the served FD prefill)      -> eval/released-dev500-routedprompt     (paired vs released-dev500)
#               the backbone: every layer, dense prompt             -> eval/base-densedecode-dev500          (paired vs released-dev500)
#   ckpt RUN ITER... : runs/RUN/ckpt_itNNN for each ITER, routed prompt paired vs the backbone (the owner's length target is defined
#               there) and dense prompt paired vs released          -> eval/RUN-itNNN-dev500-routedprompt, eval/RUN-itNNN-dev500
# Refuses while a trainer runs (one job on the node's four GPUs). An evaluation with a summary.json is not repeated. Dev rows only: the
# test split is not on the node. DRY_RUN=1 prints the commands.  usage: bash cloudlab_dev_eval.sh baselines
#                                                                         bash cloudlab_dev_eval.sh ckpt skipret-B4-20261115 012 024
set -uo pipefail
MODE=${1:?baselines|ckpt}; shift; DRY=${DRY_RUN:-}
O=/mydata/fdfix; V=$O/venv/bin; M=$O/models/FlexiDepth-Llama-3-8B-Instruct; T=$O/tools; DA=$O/data; PT=$O/serving/flexidepth_router_weights.pt
EV=$O/eval; mkdir -p "$EV"
for f in "$M/.verified" "$PT" "$DA/pool.json" "$DA/gsm8k-rows/train.jsonl" "$T/fdfix_eval.py"; do [ -e "$f" ] || { echo "missing $f"; exit 2; }; done
[ ! -e "$DA/gsm8k-rows/test.jsonl" ] || { echo "a test split is present on the node: refusing"; exit 3; }
# one evaluation invocation at a time on the node (an exclusive lock held until this script exits)
exec 9> "$EV/.lock"; flock -n 9 || { echo "another cloudlab_dev_eval.sh holds $EV/.lock: refusing"; exit 5; }
# no trainer: its launcher "$V/python $V/torchrun ... $T/fdfix_grpo.py ..." or its ranks "$V/python -u $T/fdfix_grpo.py ...", anchored so a
# shell whose own command line names the trainer does not match; a pgrep error refuses as well
notrainer(){ pgrep -f "^$V/python (-u $T/fdfix_grpo\.py|$V/torchrun .* $T/fdfix_grpo\.py)( |$)" >/dev/null; local rc=$?
  [ $rc -eq 1 ] || { [ $rc -eq 0 ] && echo "a trainer is running: refusing (it holds the GPUs)" || echo "pgrep failed (rc=$rc): refusing"; exit 4; }; }
notrainer                                                                     # checked again before every evaluation (ev)
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HOME="$EV/hf_home"
FAIL=0
ev(){ local out=$1; shift; [ -f "$out/summary.json" ] && { echo "done already: $(basename "$out")"; return 0; }
  notrainer
  local cmd=("$V/torchrun" --standalone --nproc_per_node 4 "$T/fdfix_eval.py" --released-pt "$PT" --rows "$DA/gsm8k-rows" --pool "$DA/pool.json"
             --model "$M" --out "$out" "$@")
  if [ -n "$DRY" ]; then printf '%q ' "${cmd[@]}"; echo; return 0; fi
  echo "[$(date -u +%FT%TZ)] start $(basename "$out")" >> "$EV/status.txt"
  HF_MODULES_CACHE="$EV/hf_modules" "${cmd[@]}" >> "$out.log" 2>&1; local rc=$?
  echo "[$(date -u +%FT%TZ)] $(basename "$out") rc=$rc" >> "$EV/status.txt"; [ $rc -eq 0 ] || FAIL=$rc; }
case $MODE in
  baselines)
    ev "$EV/released-dev500" --label released
    [ -n "$DRY" ] || [ -f "$EV/released-dev500/summary.json" ] || { echo "released-dev500 failed: the two paired baselines are not run"; exit 6; }
    ev "$EV/released-dev500-routedprompt" --routed-prompt --baseline "$EV/released-dev500" --label released-dev500-routedprompt
    ev "$EV/base-densedecode-dev500" --dense-decode --baseline "$EV/released-dev500" --label base-densedecode-dev500;;
  ckpt)
    RUN=${1:?run directory name under runs/}; shift; [ $# -gt 0 ] || { echo "no iterations"; exit 2; }
    [[ $RUN =~ ^skipret-[A-Za-z0-9]+-[A-Za-z0-9]+$ ]] || { echo "bad run name $RUN"; exit 2; }
    for b in released-dev500 base-densedecode-dev500; do [ -n "$DRY" ] || [ -f "$EV/$b/summary.json" ] || { echo "run 'baselines' first ($b)"; exit 2; }; done
    for k in "$@"; do
      [[ $k =~ ^[0-9]{3}$ ]] || { echo "bad iteration $k (three digits)"; exit 2; }
      RW=$O/runs/$RUN/ckpt_it$k/router_weights_vskipper_format.pt; [ -n "$DRY" ] || [ -f "$RW" ] || { echo "missing $RW"; FAIL=2; continue; }
      ev "$EV/$RUN-it$k-dev500-routedprompt" --router-pt "$RW" --routed-prompt --baseline "$EV/base-densedecode-dev500" --label "$RUN-it$k-routedprompt"
      ev "$EV/$RUN-it$k-dev500" --router-pt "$RW" --baseline "$EV/released-dev500" --label "$RUN-it$k"
    done;;
  *) echo "unknown mode $MODE"; exit 2;;
esac
exit $FAIL
