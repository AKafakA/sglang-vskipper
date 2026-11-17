#!/usr/bin/env bash
# b4_segments2.sh -- b4_segments.sh with the review fixes. Unchanged: the
# segments (060, 072, 084, 096), cloudlab_skipret_run.sh's training arguments, the node-local dev checks after every segment and the stop
# after a segment whose routed-prompt dev mean is at or below 112 tokens. Fixed:
#  - the stop value is read fail-closed: a missing summary, a missing or non-finite "tokens" stops with exit 5 (b4_segments.sh went on);
#  - a pgrep error counts as busy, not idle;
#  - GPU memory gates every launch and every dev check (all four GPUs below 1,024 MiB), so a GPU user that matches no process name is seen;
#    the busy check runs again right before the launch (waits) and before the dev checks;
#  - one orchestrator at a time (flock on /mydata/fdfix/b4_segments.lock);
#  - --adopt NNN takes over a segment an earlier orchestrator launched: wait for that trainer, then require ckpt_itNNN/state.pt (exit 3).
# usage (on the node, from this directory): nohup bash b4_segments2.sh [--adopt NNN] > b4_segments2_<UTC>.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")"
FD=/mydata/fdfix
R=$FD/runs/skipret-B4-20261115; EV=$FD/eval
ADOPT=""
if [ $# -gt 0 ]; then
  [ "$1" = --adopt ] && [[ ${2:-} =~ ^(060|072|084|096)$ ]] || { echo "usage: b4_segments2.sh [--adopt 060|072|084|096]"; exit 2; }
  ADOPT=$2
fi
exec 9> "$FD/b4_segments.lock"; flock -n 9 || { echo "another b4_segments2 holds $FD/b4_segments.lock"; exit 6; }
log(){ echo "[$(date -u +%FT%TZ)] $*"; }
pg(){ pgrep -f "$1" >/dev/null; local rc=$?; [ $rc -eq 1 ] && return 1; [ $rc -eq 0 ] || log "pgrep rc=$rc: counted as busy"; return 0; }
PY="^$FD/venv/bin/python( -u)? ($FD/venv/bin/torchrun .*)?$FD/tools/fdfix_"
trainer(){ pg "${PY}grpo\.py"; }
gpu_used(){ local m; m=$(timeout 60 nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null) || return 0
  [ "$(grep -c . <<<"$m")" -eq 4 ] || return 0; awk '$1 + 0 >= 1024 {u = 1} END {exit !u}' <<<"$m"; }
busy(){ pg "^bash cloudlab_dev_eval\.sh" || pg "${PY}(eval|grpo)\.py" || pg "^$FD/venv/bin/python $FD/venv/bin/torchrun " || gpu_used; }
wait_free(){ local n=0; while busy; do n=$((n + 1)); [ $n -ge 90 ] && [ $((n % 60)) -eq 30 ] && log "still waiting for free GPUs ($((n / 2)) min)"; sleep 30; done; }
log "start (adopt ${ADOPT:-none})"
for T in 060 072 084 096; do
  wait_free
  if [ "$T" = "$ADOPT" ]; then
    [ -f $R/ckpt_it$T/state.pt ] || { log "adopted segment it$T: ckpt_it$T/state.pt missing after its trainer ended: stop"; exit 3; }
    log "adopted segment it$T: trained"
  elif [ ! -d $R/ckpt_it$T ]; then
    last=$(ls -d $R/ckpt_it[0-9][0-9][0-9] | sort | tail -1)
    log "segment to it$T from $(basename "$last"); GPUs $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | tr '\n' ' ')"
    busy && { log "GPUs busy again right before the launch: waiting"; wait_free; }
    bash cloudlab_skipret_run.sh B4 $((10#$T)) || { log "launch failed: stop"; exit 2; }
    sleep 90
    while trainer; do sleep 60; done
    [ -f $R/ckpt_it$T/state.pt ] || { log "ckpt_it$T missing after training: stop"; exit 3; }
    log "trained to it$T"
  fi
  wait_free
  bash cloudlab_dev_eval.sh ckpt skipret-B4-20261115 $T; rc=$?; log "dev it$T rc=$rc"
  [ $rc -eq 0 ] || { log "dev failed: stop"; exit 4; }
  tok=$(python3 -I - "$EV/skipret-B4-20261115-it$T-dev500-routedprompt/summary.json" <<'EOF'
import json, math, sys
v = json.load(open(sys.argv[1]))["tokens"]
if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
    sys.exit(f"tokens is not a finite number: {v!r}")
print(v)
EOF
) || { log "cannot read the routed dev mean for it$T: stop"; exit 5; }
  log "it$T routed dev mean tokens $tok"
  python3 -I -c "import sys; sys.exit(0 if float(sys.argv[1]) <= 112.0 else 1)" "$tok" && { log "routed dev mean <= 112 at it$T: stopping for the owner"; exit 0; }
done
log "reached it096"
