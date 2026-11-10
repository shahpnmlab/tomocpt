#!/usr/bin/env bash
# Gate 8c.1 (G8c-C9, 8c.1 part): 300-step loss overlay, chunk files vs on-the-fly crops, CZII fold 0,
# recipe czii_3b_ema_lr. Two seeds per arm: the chunk arm's seed-to-seed spread is the noise.
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT=${OUT:-benchmarks/out/perf_8c_$(date +%Y%m%d)}
MAIN=/home/pranav/Desktop/pranav/software/tomocpt
CZII=$MAIN/benchmarks/out/czii_3b_ema_lr/config.yaml
CH=$MAIN/data/czii/chunks_v2_exact_fold0
ST=$MAIN/data/czii/otf_v2_exact_fold0
mkdir -p "$OUT"
run() { echo "== $1"; local n=$1; shift; PYTHONPATH=$PWD python -W ignore -m benchmarks.perf.step --config "$CZII" \
  --steps 300 --warmup 10 --workers 8 "$@" --out "$OUT/$n.json" 2>&1 | grep -vE "Warning|warn\(|^\s*$" | tail -1; }
for S in 0 1; do
  run overlay_chunks_s$S --data chunks:$CH --seed $S
  run overlay_otf_s$S --data chunks:$ST --seed $S --set train.sampling=on_the_fly --set train.otf.seed=$((42 + S))
done
