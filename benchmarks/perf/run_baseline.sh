#!/usr/bin/env bash
# Gate 8a baseline: step micro-benchmarks (real CZII / filament recipes, synthetic C=134), network
# profile, loader stages. Run from the perf worktree; GPU must be free.
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT=${OUT:-benchmarks/out/perf_baseline_$(date +%Y%m%d)}
MAIN=/home/pranav/Desktop/pranav/software/tomocpt
CZII=${CZII:-$MAIN/benchmarks/out/czii_3b_ema_lr/config.yaml}
FIL=${FIL:-$MAIN/benchmarks/out/filaments_3c2_membrane_contrastive_20261108/filaments/config.yaml}
STEPS=${STEPS:-300}; WARM=${WARM:-50}
mkdir -p "$OUT"
run() { echo "== $1"; shift; python -W ignore -m "$@" 2>&1 | grep -vE "Warning|warn\(|^\s*$" | tail -3; }
run czii_step benchmarks.perf.step --config "$CZII" --data chunks --batch 16 --steps $STEPS --warmup $WARM --profile --out "$OUT/czii_step.json"
run fil_step benchmarks.perf.step --config "$FIL" --data chunks --batch 16 --steps $STEPS --warmup $WARM --out "$OUT/fil_step.json"
for B in 4 8 16; do
  for HEAD in sigmoid softmax; do
    run c134_b${B}_$HEAD benchmarks.perf.step --data synthetic --classes 134 --batch $B --steps 20 --warmup 5 \
        --set train.particles.class_head=$HEAD --out "$OUT/synth_c134_b${B}_${HEAD}.json" || echo "failed (OOM?)"
  done
done
run czii_loader benchmarks.perf.loader --config "$CZII" --workers 0 4 8 16 --batches 30 --out "$OUT/czii_loader.json"
run fil_loader benchmarks.perf.loader --config "$FIL" --workers 0 4 8 16 --batches 30 --out "$OUT/fil_loader.json"
echo done
