#!/usr/bin/env bash
# k-fold CV on CZII with the built-in CV mode (Gate 4c; earlier versions looped the folds here).
#
# `tomocpt train` with train.cv_all_folds=True chunks every tomogram once (<OUT>/chunks/pool),
# trains each fold in its own process (<OUT>/cv/folds/fold<k>), builds the fold ensemble with
# out-of-fold calibration (<OUT>/cv/ensemble.json, oof_picks.npz) and writes the CV report
# (<OUT>/cv/cv_summary.md, cv_summary.json, peak_profiles.png).
#
# Usage (from anywhere):
#   nohup experiments/run_cv.sh > cv.log 2>&1 &
#
# Settings (environment variables, defaults in brackets):
#   CONFIG [data/czii/v2_config.yaml]  base config (prepData, train, infer sections)
#   OUT [data/czii/cv_<date>]          reuse an OUT to resume: finished folds are skipped
#   FOLDS [5]  EPOCHS [60]  CLASS_HEAD [softmax]  CROSS_CLASS_K [0.0]
#   WEIGHTS [CZII Kaggle weights]      train.cv_class_weights of the report
#   PREDICT_DIR                        optional: tomograms to predict with the ensemble
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
source /home/pranav/Desktop/pranav/software/miniconda3/etc/profile.d/conda.sh
conda activate tomocpt

CONFIG="${CONFIG:-$REPO/data/czii/v2_config.yaml}"
OUT="${OUT:-$REPO/data/czii/cv_$(date +%Y%m%d_%H%M)}"
FOLDS="${FOLDS:-5}"
EPOCHS="${EPOCHS:-60}"
CLASS_HEAD="${CLASS_HEAD:-softmax}"
CROSS_CLASS_K="${CROSS_CLASS_K:-0.0}"
WEIGHTS="${WEIGHTS:-beta-amylase:0,beta-galactosidase:2,thyroglobulin:2}"
mkdir -p "$OUT"
log() { echo "[$(date '+%F %T')] $*"; }

# CV config: the base config with the CV settings, chunks and model directories under OUT.
python - "$CONFIG" "$OUT/config.yaml" "$FOLDS" "$OUT" "$EPOCHS" "$CLASS_HEAD" "$CROSS_CLASS_K" "$WEIGHTS" <<'PY'
import sys, yaml
src, dst, folds, out, epochs, head, kneg, weights = sys.argv[1:]
cfg = yaml.safe_load(open(src))
for sec in ("shared", "train"):
    cfg.setdefault(sec, {})["chunks_dir"] = f"{out}/chunks"
t = cfg["train"]
t.update(model_dir=out, experiment_name="cv", n_epochs=int(epochs), cv_folds=int(folds), cv_all_folds=True,
         cv_class_weights=weights, data_pipeline_version=2)
t.setdefault("particles", {}).update(class_head=head, cross_class_neg_weight=float(kneg))
yaml.safe_dump(cfg, open(dst, "w"), sort_keys=False)
PY

log "k-fold CV: $FOLDS folds (log: $OUT/train.log)"
tomocpt train --config_file "$OUT/config.yaml" > "$OUT/train.log" 2>&1

{ echo; echo "## k-fold CV on CZII ($FOLDS folds, $EPOCHS epochs, $CLASS_HEAD head): \`$OUT\`"; echo
  cat "$OUT/cv/cv_summary.md"; } >> experiments/czii_v2_results.md
log "results appended to experiments/czii_v2_results.md"

if [[ -n "${PREDICT_DIR:-}" ]]; then
    log "predicting $PREDICT_DIR with the ensemble -> $OUT/predictions"
    tomocpt predict --config_file "$CONFIG" --tomogram_dir "$PREDICT_DIR" --weights "$OUT/cv/ensemble.json" \
        --predictions_dir "$OUT/predictions" > "$OUT/predict.log" 2>&1
fi
log "done: $OUT"
