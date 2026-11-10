#!/usr/bin/env bash
# Gate 8c acceptance (human decision 10 Nov 2026): two CZII 5-fold CVs with recipe czii_3b_ema_lr,
# built-in CV mode, same seeds, run from this worktree (PYTHONPATH; fold processes inherit it):
#   chunks : grid chunks (the baseline; no 5-fold CV of this recipe existed)
#   otf    : train.sampling=on_the_fly + train.otf.augment_device=cuda (8c.1 + 8c.2)
# Do not edit this worktree while it runs: every fold starts a fresh process.
set -uo pipefail
cd "$(dirname "$0")/../.."
WT=$PWD
export PYTHONPATH=$WT
PY=/home/pranav/Desktop/pranav/software/miniconda3/envs/tomocpt/bin/python
MAIN=/home/pranav/Desktop/pranav/software/tomocpt
BASE=$MAIN/benchmarks/out/czii_3b_ema_lr/config.yaml
ROOT=${ROOT:-$WT/benchmarks/out/cv_8c}
WEIGHTS="beta-amylase:0,beta-galactosidase:2,thyroglobulin:2"
log() { echo "[$(date '+%F %T')] $*"; }
for arm in ${ARMS:-chunks otf}; do
  OUT=$ROOT/$arm
  mkdir -p "$OUT"
  $PY - "$BASE" "$OUT" "$arm" "$WEIGHTS" <<'PYS'
import sys, yaml
src, out, arm, weights = sys.argv[1:]
cfg = yaml.safe_load(open(src))
for sec in ("shared", "train"):
    cfg.setdefault(sec, {})["chunks_dir"] = f"{out}/chunks"
t = cfg["train"]
t.update(model_dir=out, experiment_name="cv", cv_folds=5, cv_all_folds=True, cv_class_weights=weights)
t.pop("cv_fold", None)
if arm == "otf":
    t["sampling"] = "on_the_fly"
    t["otf"] = {"augment_device": "cuda"}
yaml.safe_dump(cfg, open(f"{out}/config.yaml", "w"), sort_keys=False)
PYS
  log "arm $arm start ($(git -C "$WT" rev-parse --short HEAD))"
  $PY -W ignore -m tomocpt.main train --config_file "$OUT/config.yaml" > "$OUT/train.log" 2>&1
  log "arm $arm exit $?"
done
