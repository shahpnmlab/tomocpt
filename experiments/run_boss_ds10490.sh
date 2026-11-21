#!/usr/bin/env bash
# DS-10490 "final boss" run (human decisions 18-21 Nov 2026).
# One joint model, C = 134: 132 particle species + microtubule + actin, from scratch, at 6.8 A/px (two levels).
#
# Split: benchmarks/out/boss_20261122/split.json (seed 42): 240 train / 20 val / 40 test; the test runs come only
# from sample_23..300 (sample_1..22 were used for the filament work). sample_300 has no microtubule/actin mask and
# is dropped from training, so 239 training runs.
#
# Recipe = the Gate 3e comparison run (benchmarks/out/softmax_scope_20261115/run.sh) plus:
#   softmax_scope=particles            Gate 3e winner (filaments as independent sigmoids)
#   Fourier + missing-wedge aug        on (pipeline-v2 default since the gpu-augment merge; also set explicitly)
#   lr_warmup_steps=1024               Gate 9
#   contrastive_warmup_steps=1024      Gate 9 (also ramps the embedding CE)
#   bank_max_age=5000                  Gate 9
#   repeat_factor_threshold=1.0, repeat_factor_max=10   Gate 9 (check the logged factors in train.log first)
#   embed_dim=128                      Gate 9
#   embed_ce_weight=0.1, prototype_head=true   Gate 9b, options 3 + 2 (ablation: a tie on fine-tunes; human choice)
#   contrastive_weight=0.1             unchanged (lambda 0.5 hurt microtubules in the ablation)
# Batch 16, 16384 crops per epoch (1024 steps, ~1.1-1.2 h), early stopping (driver recipe: relative 0.007,
# patience 10), at most 100 epochs (~4.5-5 days).
#
# Centrelines: data/filaments/10490/centerlines_all/<object>/<run>.mod (links to centerlines/ for sample_1..22 and
# centerlines_23_299/ from scripts/masks_to_centerlines.py, default templates, as for the first 22).
#
# The driver evaluates on the 20 validation runs (thresholds calibrated on the same runs). The 40 test runs are
# scored afterwards by a separate predict-and-score step (validation-calibrated thresholds).
#
# Usage: bash experiments/run_boss_ds10490.sh            # full run
#        SMOKE=1 bash experiments/run_boss_ds10490.sh    # ~300 steps on 4 train + 1 val run, same settings
set -u
R=$(cd "$(dirname "$0")/.." && pwd)
PY=/home/pranav/Desktop/pranav/software/miniconda3/envs/tomocpt/bin/python
DATA=$R/data/filaments
SRC=$R/benchmarks/out/softmax_scope_20261115
SPLIT=$R/benchmarks/out/boss_20261122/split.json
CL=$DATA/10490/centerlines_all
cd "$R"; export PYTHONPATH=$R

TRAIN=$($PY -c "import json; s=json.load(open('$SPLIT')); print(','.join(r for r in s['train'] if r != 'sample_300'))")
VAL=$($PY -c "import json; print(','.join(json.load(open('$SPLIT'))['val']))")
EPOCHS=100; SPE=16384; O=$R/benchmarks/out/boss_20261122/run
if [ "${SMOKE:-0}" = 1 ]; then
  TRAIN=sample_1,sample_2,sample_3,sample_4; VAL=sample_6; EPOCHS=1; SPE=4800; O=$R/benchmarks/out/boss_20261122/smoke
fi

for o in microtubule actin; do
  for r in ${TRAIN//,/ } ${VAL//,/ }; do
    [ -e "$CL/$o/$r.mod" ] || { echo "missing centrelines $CL/$o/$r.mod"; exit 1; }
  done
done
mkdir -p "$O"
echo "start $(date '+%F %T') code $(git rev-parse --short HEAD) train $(echo $TRAIN | tr ',' '\n' | wc -l) val $(echo $VAL | tr ',' '\n' | wc -l)" | tee -a "$O/run.log"
$PY -W ignore benchmarks/filaments.py --data_root "$DATA" --centerlines "$CL" \
  --train_sets 10490 --test_sets "" --objects microtubule,actin --arms joint \
  --particles_star $SRC/boss_all_300runs.star --particle_classes "$(cat $SRC/particle_classes.txt)" \
  --train_runs "$TRAIN" --val_runs "$VAL" --band_max_vox 37 --epochs $EPOCHS --batch_size 16 --n_cpus 8 \
  --negative_masks membrane --contrastive_weight 0.1 --seed 0 \
  --train_set particles.softmax_scope=particles --train_set otf.samples_per_epoch=$SPE \
  --train_set augmentation.use_mw_aug=true --train_set augmentation.use_fourier_aug=true \
  --train_set lr_warmup_steps=1024 --train_set repeat_factor_threshold=1.0 --train_set repeat_factor_max=10 \
  --train_set discriminative.contrastive_warmup_steps=1024 --train_set discriminative.bank_max_age=5000 \
  --train_set discriminative.embed_dim=128 \
  --train_set discriminative.embed_ce_weight=0.1 --train_set discriminative.prototype_head=true \
  --out "$O" > "$O/driver.log" 2>&1
echo "exit $? $(date '+%F %T')" | tee -a "$O/run.log"
