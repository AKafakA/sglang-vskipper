#!/usr/bin/env bash
# cloudlab_fdfix_setup.sh -- CloudLab node bring-up for GRPO post-training and the held-out evaluator (4x A100-SXM4-40GB).
# The only way this node is set up. The system stage is the October TP-feasibility setup's (NVMe scratch, driver branch 580 from
# NVIDIA's repo, uv); the training environment is pinned package for package (fdfix-freeze.txt).
# Inputs come only from the staging folder (/mydata/fdfix/_incoming, MANIFEST.sha256): the GSM8K rows and pool, the tools.
# The model comes from the Hub at the pinned revision and is checked against the shard sha256 list; the released router file is
# rebuilt from it and must hash to the paper's file. Idempotent; fail-closed.
#
# usage (on the node):  bash cloudlab_fdfix_setup.sh system     # may exit 75 = reboot required, then re-run
#                       bash cloudlab_fdfix_setup.sh env        # venv from the pinned freeze
#                       bash cloudlab_fdfix_setup.sh inputs     # model, router file, data, tools (needs _incoming)
#                       bash cloudlab_fdfix_setup.sh verify
#                       DRY_RUN=1 bash cloudlab_fdfix_setup.sh <stage>   # print the commands, change nothing
set -euo pipefail
STAGE=${1:?system|env|inputs|verify}
MNT=/mydata; NVME=/dev/nvme0n1; LABEL=vpipe-scratch
O=$MNT/fdfix; IN=$O/_incoming; LOG=$O/setup.log
FD_REPO=xuan-luo/FlexiDepth-Llama-3-8B-Instruct; FD_REV=2ce73595ad0467fedd539c113e5b9deed046df32
ROUTER_SHA=9a1759073d4b68ad24f249f5c303a7d74c9b51b0289a6e96812dcb5d3ffed6b9
MODELING_SHA=a7a1ced8e3ee2fcfdaf2b08df3f2661e754cbf754f6decf2bfc2eaeb5c966b0f   # modeling_ddllama.py at 2ce73595 (Hub fetch 2026-11-14)
DRIVER_BRANCH=580
DRY=${DRY_RUN:-}
log(){ local m="[$(date -u +%FT%TZ)] [$STAGE] $*"; echo "$m"; [ -n "$DRY" ] || { [ -d "$O" ] && echo "$m" >> "$LOG"; } || true; }
die(){ log "FATAL: $*"; exit 1; }
run(){ if [ -n "$DRY" ]; then echo "+ $*"; else "$@"; fi; }
export PATH=$HOME/.local/bin:$PATH

stage_system(){
  # scratch: format the NVMe only when it is verifiably empty; afterwards mount it by its label (survives reboots)
  if mountpoint -q $MNT; then log "$MNT mounted"; else
    local dev; dev=$(readlink -f $NVME); [[ $dev == /dev/nvme* ]] || die "$NVME resolved to $dev"
    if sudo blkid -L $LABEL >/dev/null 2>&1; then
      run sudo mkdir -p $MNT; run sudo mount -L $LABEL $MNT; log "mounted existing $LABEL"
    else
      [ -z "$(sudo blkid "$dev" 2>/dev/null)" ] && [ "$(lsblk -no NAME "$dev" | tail -n +2 | wc -l)" -eq 0 ] \
        || die "refusing to format $dev: it carries a signature or partitions"
      log "formatting $dev (verified empty)"; run sudo mkfs.ext4 -q -F -m 0 -L $LABEL "$dev"
      run sudo mkdir -p $MNT; run sudo mount -L $LABEL $MNT
    fi
    run sudo chown "$USER:$(id -gn)" $MNT
  fi
  grep -q "LABEL=$LABEL" /etc/fstab || run sudo sh -c "echo 'LABEL=$LABEL $MNT ext4 defaults,nofail 0 2' >> /etc/fstab"
  run mkdir -p $O/{_incoming,models,serving,data,tools,runs,logs}
  export DEBIAN_FRONTEND=noninteractive
  # driver branch 580, as in October (the newest branch segfaulted in the serving stack's graph instantiation; the training lanes
  # need only a driver new enough for CUDA 12.8 wheels, so the same pin serves both)
  local other; other=$(dpkg-query -W -f='${Package} ${Version}\n' 2>/dev/null \
    | awk -v b="$DRIVER_BRANCH." '$2 ~ /^[0-9][0-9][0-9]\.[0-9]+\.[0-9]+-/ && index($2, b) != 1 {print $1}' \
    | { grep -vxE 'nvidia-modprobe|nvidia-persistenced|nvidia-settings|libxnvctrl0' || true; } | tr '\n' ' ')
  if [ -n "${other// }" ]; then
    log "driver packages of another branch present ($other); pinning $DRIVER_BRANCH"
    run sudo dpkg --purge --force-depends $other
    run sudo -E apt-get install -y -qq -f --allow-downgrades --allow-change-held-packages
    run sudo -E apt-get install -y -qq --allow-downgrades --allow-change-held-packages cuda-drivers-$DRIVER_BRANCH
    log "driver $DRIVER_BRANCH installed: sudo reboot, then re-run 'system'"; [ -n "$DRY" ] || exit 75
  fi
  run sudo -E apt-get update -qq
  run sudo -E apt-get install -y -qq build-essential git curl ca-certificates jq zstd
  if ! nvidia-smi >/dev/null 2>&1; then
    if [ ! -f /usr/share/keyrings/cuda-archive-keyring.gpg ]; then
      run curl -fsSL -o /tmp/cuda-keyring.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
      run sudo dpkg -i /tmp/cuda-keyring.deb; run sudo -E apt-get update -qq
    fi
    run sudo -E apt-get install -y -qq --allow-change-held-packages dkms
    run sudo -E apt-get install -y -qq --allow-change-held-packages cuda-drivers-$DRIVER_BRANCH
    run sudo modprobe nvidia || true
  fi
  if [ -z "$DRY" ] && ! nvidia-smi >/dev/null 2>&1; then log "driver installed, not loaded: sudo reboot, then re-run 'system'"; exit 75; fi
  [ -n "$DRY" ] || { nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader | tee -a $LOG; }
  if [ -z "$DRY" ]; then
    local drv; drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
    [[ $drv == $DRIVER_BRANCH.* ]] || die "running driver $drv is not branch $DRIVER_BRANCH"
  fi
  command -v uv >/dev/null 2>&1 || run sh -c "curl -LsSf https://astral.sh/uv/install.sh | sh"
  log "system OK"
}

stage_env(){
  [ -f $IN/fdfix-freeze.txt ] && [ -f $IN/MANIFEST.sha256 ] || die "no shipped freeze or MANIFEST"
  [ -n "$DRY" ] || (cd $IN && sha256sum -c --quiet MANIFEST.sha256) || die "shipped inputs fail their sha256 check"
  local fh; fh=$(sha256sum $IN/fdfix-freeze.txt | cut -d' ' -f1)
  if [ "$(cat $O/venv/.ready 2>/dev/null)" != "$fh" ]; then
    [ -e $O/venv ] && run mv $O/venv $O/venv.void-$(date -u +%Y%m%dT%H%M%SZ)
    run uv venv -q --python 3.12 $O/venv
    # the pinned training environment, package for package; torch from the cu128 index
    VIRTUAL_ENV=$O/venv run uv pip install -q --torch-backend cu128 -r $IN/fdfix-freeze.txt
    [ -n "$DRY" ] || echo "$fh" > $O/venv/.ready                          # readiness bound to this freeze
  fi
  [ -n "$DRY" ] || $O/venv/bin/python -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__, 'cuda', torch.version.cuda)" | tee -a $LOG
  log "env OK"
}

stage_inputs(){
  [ -f $IN/MANIFEST.sha256 ] || die "no $IN/MANIFEST.sha256 (ship_fdfix.sh)"
  [ -n "$DRY" ] || (cd $IN && sha256sum -c --quiet MANIFEST.sha256) || die "shipped inputs fail their sha256 check"
  run rsync -a $IN/data/ $O/data/; run rsync -a $IN/tools/ $O/tools/
  local F=$O/models/FlexiDepth-Llama-3-8B-Instruct
  if [ ! -f $F/.verified ]; then
    run $O/venv/bin/python -c "from huggingface_hub import snapshot_download as d; d('$FD_REPO', revision='$FD_REV', local_dir='$F')"
    # every safetensors shard against the Hub's LFS sha256 list recorded on 2026-11-14 (data/shard_oids_2ce73595.json)
    [ -n "$DRY" ] || MODELING_SHA=$MODELING_SHA $O/venv/bin/python - "$F" "$O/data/shard_oids_2ce73595.json" <<'PY' || die "model files fail their sha256 check"
import hashlib, json, os, sys
d, ref = sys.argv[1], json.load(open(sys.argv[2]))
oids = {k: v for k, v in ref.items() if v}                          # the four shards; config.json and the code carry no LFS oid
assert len(oids) == 4, oids
oids["modeling_ddllama.py"] = os.environ["MODELING_SHA"]
bad = []
for name, oid in oids.items():
    h = hashlib.sha256()
    with open(os.path.join(d, name), "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    if h.hexdigest() != oid:
        bad.append(name)
print("shards checked:", len(oids), "bad:", bad); sys.exit(1 if bad else 0)
PY
    run touch $F/.verified
  fi
  # the paper's router file, staged and covered by MANIFEST.sha256 (not rebuilt: another torch may not reproduce its bytes)
  local RW=$O/serving/flexidepth_router_weights.pt
  run mkdir -p $O/serving; run cp $IN/serving/flexidepth_router_weights.pt $RW.tmp; run mv $RW.tmp $RW
  [ -n "$DRY" ] || [ "$(sha256sum $RW | cut -d' ' -f1)" = $ROUTER_SHA ] || die "router file sha256 differs from the paper's file"
  log "inputs OK (model $FD_REV verified, router file = paper's)"
}

stage_verify(){
  nvidia-smi --query-gpu=index,name,memory.total,driver_version,power.limit --format=csv,noheader | tee -a $LOG
  $O/venv/bin/python -c "import torch; print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1), 'GiB')" | tee -a $LOG
  ls -la $O/models/FlexiDepth-Llama-3-8B-Instruct/.verified $O/serving/flexidepth_router_weights.pt $O/data $O/tools | tee -a $LOG
}

case $STAGE in
  system) stage_system;; env) stage_env;; inputs) stage_inputs;; verify) stage_verify;;
  *) die "unknown stage $STAGE";;
esac
