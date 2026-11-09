#!/usr/bin/env bash
# Gate 8a baseline of predict, calibration and chunking (GPU must be free).
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT=${OUT:-benchmarks/out/perf_baseline_$(date +%Y%m%d)}
MAIN=/home/pranav/Desktop/pranav/software/tomocpt
CZ_CKPT=$MAIN/benchmarks/out/czii_3b_ema_lr/model/fold0/checkpoints/weights.ckpt
CZ_CH=$MAIN/data/czii/chunks_v2_exact_fold0
FD=$MAIN/benchmarks/out/filaments_3c2_membrane_contrastive_20261108/filaments
mkdir -p "$OUT"
run() { echo "== $1"; shift; python -W ignore -m benchmarks.perf.pipeline "$@" 2>&1 | grep -vE "Warning|warn\(|^\s*$|INFO" | tail -14; }
run czii_predict predict --ckpt $CZ_CKPT --chunks_dir $CZ_CH --n 2 --profile --out "$OUT/czii_predict.json"
run fil_predict predict --ckpt $FD/model/bench/checkpoints/weights.ckpt --chunks_dir $FD/chunks --n 2 --profile --out "$OUT/fil_predict.json"
run czii_calibrate calibrate --ckpt $CZ_CKPT --chunks_dir $CZ_CH --profile --out "$OUT/czii_calibrate.json"
run fil_calibrate calibrate --ckpt $FD/model/bench/checkpoints/weights.ckpt --chunks_dir $FD/chunks --profile --out "$OUT/fil_calibrate.json"
run czii_chunk chunk --config $MAIN/benchmarks/out/czii_3b_ema_lr/config.yaml --tmp "$OUT/tmp_chunks" --out "$OUT/czii_chunk.json"
echo done
