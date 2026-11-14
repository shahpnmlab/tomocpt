#!/usr/bin/env bash
# Gate 8c.2 (G8c-C9, 8c.2 part): data wait with augment_device=cuda at C=6 (CZII fold 0) and
# C=134 (the CZII volumes with 134 synthetic particle classes, ~3000 points per tomogram), B=4,
# recipe czii_3b_ema_lr; the CPU path at C=6 for reference. C134 = path of that store
# (built by make_c134_store.py, outside the repository).
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT=${OUT:-benchmarks/out/perf_8c_$(date +%Y%m%d)}
MAIN=/home/pranav/Desktop/pranav/software/tomocpt
CZII=$MAIN/benchmarks/out/czii_3b_ema_lr/config.yaml
ST=$MAIN/data/czii/otf_v2_exact_fold0
C134=${C134:?set C134 to the 134-class store}
mkdir -p "$OUT"
run() { echo "== $1"; local n=$1; shift; PYTHONPATH=$PWD python -W ignore -m benchmarks.perf.step --config "$CZII" \
  --steps 150 --warmup 30 --batch 4 "$@" --set train.sampling=on_the_fly --out "$OUT/$n.json" 2>&1 \
  | grep -vE "Warning|warn\(|^\s*$" | tail -1; }
run wait_c6_cuda  --data chunks:$ST   --workers 2 --set train.otf.augment_device=cuda
run wait_c6_cpu   --data chunks:$ST   --workers 8 --set train.otf.augment_device=cpu
D=$(python -c "import json; t=json.load(open('$C134/chunk_layout.json'))['class_table']; print(','.join(f\"{r['number']}:{r['diameter_ang']:.3f}\" for r in t))")
N=$(python -c "print(','.join(f'{i+1}:c{i}' for i in range(134)))")
run wait_c134_cuda --data chunks:$C134 --workers 2 --set train.otf.augment_device=cuda \
  --set "prepData.class_diameters=$D" --set "prepData.class_names=$N"
# overlay of the GPU path (seeds 0, 1; compare with run_8c1_overlay.sh outputs), B 16
for S in 0 1; do
  PYTHONPATH=$PWD python -W ignore -m benchmarks.perf.step --config "$CZII" --data chunks:$ST --steps 300 --warmup 10 \
    --workers 2 --seed $S --set train.sampling=on_the_fly --set train.otf.seed=$((42 + S)) \
    --set train.otf.augment_device=cuda --out "$OUT/overlay_otfgpu_s$S.json" 2>&1 | grep -vE "Warning|warn\(|^\s*$" | tail -1
done
