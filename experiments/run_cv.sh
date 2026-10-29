#!/usr/bin/env bash
# k-fold training, out-of-fold (OOF) threshold calibration and fold ensemble (Gate 4b), CZII.
#
# Each fold trains a pipeline-v2 picker with one fold of tomograms held out for validation
# (its own chunks_dir and model_dir). The fold models' picks on their own held-out tomograms
# are pooled to calibrate one threshold per class; every tomogram is then scored by a model
# that never trained on it (the cross-validation estimate). The fold models form the ensemble.
#
# Usage (from anywhere):
#   nohup experiments/run_cv.sh > cv.log 2>&1 &
#
# Settings (environment variables, defaults in brackets):
#   CONFIG [data/czii/v2_config.yaml]  base config (prepData, train, infer sections)
#   OUT [data/czii/cv_<date>]          reuse an OUT to resume: finished folds are skipped
#   FOLDS [5]  EPOCHS [60]  CLASS_HEAD [softmax]  CROSS_CLASS_K [0.0]
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
mkdir -p "$OUT"
log() { echo "[$(date '+%F %T')] $*"; }

ckpts=()
for ((k = 0; k < FOLDS; k++)); do
    fold="$OUT/fold$k"
    mkdir -p "$fold"
    ckpt="$(find "$fold/model" -name '*.ckpt' 2>/dev/null | sort | tail -1 || true)"
    if [[ -f "$fold/done" && -n "$ckpt" ]]; then
        log "fold $k already trained: $ckpt"
    else
        # Per-fold config: k-fold split, own chunks and model directories, class head.
        python - "$CONFIG" "$fold/config.yaml" "$FOLDS" "$k" "$fold" "$EPOCHS" "$CLASS_HEAD" "$CROSS_CLASS_K" <<'PY'
import sys, yaml
src, dst, folds, k, fold, epochs, head, kneg = sys.argv[1:]
cfg = yaml.safe_load(open(src))
for sec in ("shared", "train"):
    cfg.setdefault(sec, {})["chunks_dir"] = f"{fold}/chunks"
t = cfg["train"]
t.update(model_dir=f"{fold}/model", experiment_name=f"fold{k}", n_epochs=int(epochs),
         cv_folds=int(folds), cv_fold=int(k), data_pipeline_version=2)
t.setdefault("particles", {}).update(class_head=head, cross_class_neg_weight=float(kneg))
yaml.safe_dump(cfg, open(dst, "w"), sort_keys=False)
PY
        log "fold $k / $FOLDS: training (log: $fold/train.log)"
        tomocpt train --config_file "$fold/config.yaml" > "$fold/train.log" 2>&1
        ckpt="$(find "$fold/model" -name '*.ckpt' | sort | tail -1)"
        touch "$fold/done"
    fi
    ckpts+=("$ckpt")
done

log "building the ensemble from out-of-fold picks"
python -m tomocpt.predict.ensemble "${ckpts[@]}" --out "$OUT/ensemble.json" --config_file "$CONFIG" \
    > "$OUT/ensemble.log" 2>&1

# Cross-validation estimate: per-class F4 of the pooled OOF picks at the calibrated thresholds,
# and the Kaggle-weighted mean (weights 0 beta-amylase, 2 beta-galactosidase and thyroglobulin, 1 others).
python - "$OUT/ensemble.json" "$CONFIG" "$OUT" "$FOLDS" "$EPOCHS" "$CLASS_HEAD" <<'PY' | tee "$OUT/cv_summary.md"
import json, sys, yaml
path, config, out, folds, epochs, head = sys.argv[1:]
cal = json.load(open(path))["calibration"]
names = dict(p.split(":", 1) for p in yaml.safe_load(open(config))["prepData"]["class_names"].split(","))
kaggle = {"beta-amylase": 0, "beta-galactosidase": 2, "thyroglobulin": 2}
rows, num, den = [], 0.0, 0.0
for number, e in sorted(cal["classes"].items(), key=lambda kv: int(kv[0])):
    name = names.get(str(number), str(number))
    thr = "-" if e["threshold"] is None else f"{e['threshold']:.2f}"
    rows.append(f"| {name} | {e['n_true']} | {thr} | {e['fbeta']:.3f} |")
    w = kaggle.get(name, 1)
    num, den = num + w * e["fbeta"], den + w
print(f"\n## k-fold CV on CZII ({folds} folds, {epochs} epochs, {head} head)\n")
print(f"- output: `{out}`; OOF tomograms: {', '.join(cal['oof_tomograms'])}")
print("- F4 of pooled out-of-fold picks at the thresholds calibrated on them (slightly optimistic)\n")
print("| class | true particles | threshold | OOF F4 |\n|---|---|---|---|")
print("\n".join(rows))
print(f"\nKaggle-weighted OOF F4: **{num / den:.3f}** (Kaggle 1st place: 0.788 on its hidden test set)\n")
PY
cat "$OUT/cv_summary.md" >> experiments/czii_v2_results.md
log "results appended to experiments/czii_v2_results.md"

if [[ -n "${PREDICT_DIR:-}" ]]; then
    log "predicting $PREDICT_DIR with the ensemble -> $OUT/predictions"
    tomocpt predict --config_file "$CONFIG" --tomogram_dir "$PREDICT_DIR" --weights "$OUT/ensemble.json" \
        --predictions_dir "$OUT/predictions" > "$OUT/predict.log" 2>&1
fi
log "done: $OUT"
