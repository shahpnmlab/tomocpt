#!/usr/bin/env bash
# Gate 9b ablation (spec .claude/agents/09b_AGENT_EMBED_HEAD.md): short fine-tunes from the DS-10490
# softmax_scope `all` checkpoint (C = 134: 132 species + microtubule + actin), same data, split and recipe
# as benchmarks/out/softmax_scope_20261115/run.sh. Arms differ only in train.discriminative:
#   a_supcon01    contrastive_weight 0.1 (as the source run)
#   b_supcon05    contrastive_weight 0.5
#   c_embed_ce    contrastive_weight 0.1 + embed_ce_weight 0.1 (option 3)
#   d_ce_proto    contrastive_weight 0.1 + embed_ce_weight 0.1 + prototype_head true (options 3 + 2; option 2 alone
#                 barely trains its prototypes while alpha ~ 0, human decision 19 Nov 2026)
# Budget: 6 epochs x 1024 steps (16384 crops / batch 16) = 6144 optimizer steps per arm. Early stopping and the
# LR plateau are pushed out (patience 100), so every arm runs exactly 6144 steps. Measured on the source run:
# 3.7 s/step + ~4 min validation per epoch -> ~6.8 h training + ~0.5 h evaluation per arm, ~29 h for all four
# (at 3 s/step: ~5.5 h + 0.5 h, ~24 h). Option 2 adds < 1 % per step (estimate, see the 9b report).
# Fine-tuning: differential LR (swinViT x0.01, encoder x0.1, decoder / out / heads / prototypes x1.0); the
# projection head and memory bank are copied from the source (Gate 9b), the prototypes start random, alpha at 0.
# Prepared labels are reused from the source run (symlink), so prepare_data does not run again.
# Run from the repository root of the branch that has Gate 9b:  bash experiments/run_embed_head_ablation.sh [arm ...]
set -u
R=$(cd "$(dirname "$0")/.." && pwd)
SRC=/home/pranav/Desktop/pranav/software/tomocpt/benchmarks/out/softmax_scope_20261115
CKPT=$SRC/all/joint/model/bench/checkpoints/weights.ckpt
O=$R/benchmarks/out/embed_head_$(date +%Y%m%d)
PY=/home/pranav/Desktop/pranav/software/miniconda3/envs/tomocpt/bin/python
DATA=/home/pranav/Desktop/pranav/software/tomocpt/data/filaments
EPOCHS=6
cd "$R"; export PYTHONPATH=$R
[ -f "$CKPT" ] || { echo "missing source checkpoint $CKPT"; exit 1; }
mkdir -p "$O"
TRAIN=sample_1,sample_2,sample_3,sample_4,sample_5$(printf ',sample_%d' $(seq 8 22))

declare -A ARM
ARM[a_supcon01]="--train_set discriminative.contrastive_weight=0.1"
ARM[b_supcon05]="--train_set discriminative.contrastive_weight=0.5"
ARM[c_embed_ce]="--train_set discriminative.contrastive_weight=0.1 --train_set discriminative.embed_ce_weight=0.1"
ARM[d_ce_proto]="--train_set discriminative.contrastive_weight=0.1 --train_set discriminative.embed_ce_weight=0.1 --train_set discriminative.prototype_head=true"
ARMS=("$@"); [ ${#ARMS[@]} -gt 0 ] || ARMS=(a_supcon01 b_supcon05 c_embed_ce d_ce_proto)

echo "start $(date '+%F %T') code $(git rev-parse --short HEAD) source $CKPT" | tee -a "$O/run.log"
for arm in "${ARMS[@]}"; do
  [ -n "${ARM[$arm]:-}" ] || { echo "unknown arm $arm"; exit 1; }
  mkdir -p "$O/$arm/joint"
  [ -e "$O/$arm/joint/training_data" ] || ln -s "$SRC/all/joint/training_data" "$O/$arm/joint/training_data"
  echo "arm $arm start $(date '+%F %T')" | tee -a "$O/run.log"
  # shellcheck disable=SC2086
  $PY -W ignore benchmarks/filaments.py --data_root $DATA --centerlines $DATA/10490/centerlines \
    --train_sets 10490 --test_sets "" --objects microtubule,actin --arms joint \
    --particles_star $SRC/all_particles.star --particle_classes "$(cat $SRC/particle_classes.txt)" \
    --train_runs $TRAIN --val_runs sample_6,sample_7 --band_max_vox 37 --epochs $EPOCHS --batch_size 16 --n_cpus 8 \
    --negative_masks membrane --seed 0 \
    --train_set particles.softmax_scope=all --train_set otf.samples_per_epoch=16384 \
    --train_set fine_tune_from=$CKPT \
    --train_set early_stop_patience=100 --train_set lr_plateau_patience=100 \
    ${ARM[$arm]} \
    --out $O/$arm > $O/$arm.log 2>&1
  echo "arm $arm exit $? $(date '+%F %T')" | tee -a "$O/run.log"
done
