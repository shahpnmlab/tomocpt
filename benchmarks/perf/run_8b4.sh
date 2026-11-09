#!/usr/bin/env bash
# Gate 8b.4 step benchmark (fused attention, current defaults), checkpointing on and off.
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT=${OUT:-benchmarks/out/perf_8b4}
MAIN=/home/pranav/Desktop/pranav/software/tomocpt
mkdir -p "$OUT"
for R in czii fil; do
  CFG=$MAIN/benchmarks/out/czii_3b_ema_lr/config.yaml
  [ $R = fil ] && CFG=$MAIN/benchmarks/out/filaments_3c2_membrane_contrastive_20261108/filaments/config.yaml
  A="--config $CFG --data chunks --batch 16 --steps 200 --warmup 40"
  python -W ignore -m benchmarks.perf.step $A --out "$OUT/${R}_defaults.json" > /dev/null 2>&1 || echo "$R defaults failed"
  python -W ignore -m benchmarks.perf.step $A --set train.network.SwinUNETR.use_checkpoint=False \
      --out "$OUT/${R}_ckpt_off.json" > /dev/null 2>&1 || echo "$R ckpt_off failed"
done
echo done
