#!/usr/bin/env bash
# Gate 8b before/after step micro-benchmarks (spec 3.3). Run from the perf worktree; GPU and CPU idle.
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT=${OUT:-benchmarks/out/perf_8b_$(date +%Y%m%d)}
MAIN=/home/pranav/Desktop/pranav/software/tomocpt
CZII=$MAIN/benchmarks/out/czii_3b_ema_lr/config.yaml
FIL=$MAIN/benchmarks/out/filaments_3c2_membrane_contrastive_20261108/filaments/config.yaml
STEPS=${STEPS:-200}; WARM=${WARM:-40}
mkdir -p "$OUT"
run() { echo "== $1"; local n=$1; shift; python -W ignore -m benchmarks.perf.step "$@" --out "$OUT/$n.json" 2>&1 | grep -vE "Warning|warn\(|^\s*$" | tail -2; }
CK="--set train.network.SwinUNETR.use_checkpoint=False"
LOSS="--set train.particles.loss_checkpoint=False --set train.particles.hessian_block=0"
PIN="--set train.pin_memory=True"
for R in czii fil; do
  CFG=$CZII; [ $R = fil ] && CFG=$FIL
  A="--config $CFG --data chunks --batch 16 --steps $STEPS --warmup $WARM"
  run ${R}_defaults $A
  run ${R}_ckpt_off $A $CK
  run ${R}_ckpt_loss $A $CK $LOSS
  run ${R}_all $A $CK $LOSS $PIN
done
for H in 1 0; do
  for LC in True False; do
    run c134_b4_sigmoid_block${H}_lossckpt${LC} --data synthetic --classes 134 --batch 4 --steps 20 --warmup 5 \
      --set train.particles.class_head=sigmoid --set train.particles.hessian_block=$H --set train.particles.loss_checkpoint=$LC
  done
done
echo done
