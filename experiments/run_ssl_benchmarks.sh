#!/usr/bin/env bash
# Gate 5 CZII benchmarks end to end: SSL pretraining (v2 and v1), picker fine-tuning from
# scratch / SSL v2 / SSL v1 with 1, 2 and all training tomograms, inference and scoring on the
# held-out tomograms, then the dense-embedding probe. Results are appended to
# experiments/czii_v2_results.md.
#
# Usage (from anywhere):
#   nohup experiments/run_ssl_benchmarks.sh > ssl_bench.log 2>&1 &
#   TINY=1 experiments/run_ssl_benchmarks.sh          # smoke run on phantoms (minutes)
#
# Settings (environment variables, defaults in brackets):
#   OUT [benchmarks/out/czii_ssl_<date>]  reuse an OUT to resume: finished steps are skipped
#   EPOCHS [60] picker epochs   SSL_V1_EPOCHS [30]  SSL_V2_EPOCHS [90] (a v2 epoch has ~3x
#   fewer samples than v1, so 90 v2 epochs ~ the optimiser steps of 30 v1 epochs)
#   SEEDS [0,1,2]  N_TRAIN [1,2,all]  ARMS [scratch,ssl_v2,ssl_v1]
#   BATCH_SIZE [8]  TEST_TOMOS [TS_5_4,TS_69_2]  SSL_V1_CLASS [4 = ribosome]
#   SSL_V2_CKPT, SSL_V1_CKPT  existing SSL checkpoints (skip pretraining)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
source /home/pranav/Desktop/pranav/software/miniconda3/etc/profile.d/conda.sh
conda activate tomocpt

CZII="$REPO/data/czii"
OUT="${OUT:-$REPO/benchmarks/out/czii_ssl_$(date +%Y%m%d_%H%M)}"
EPOCHS="${EPOCHS:-60}"
SSL_V1_EPOCHS="${SSL_V1_EPOCHS:-30}"
SSL_V2_EPOCHS="${SSL_V2_EPOCHS:-90}"
SEEDS="${SEEDS:-0,1,2}"
N_TRAIN="${N_TRAIN:-1,2,all}"
ARMS="${ARMS:-scratch,ssl_v2,ssl_v1}"
BATCH_SIZE="${BATCH_SIZE:-8}"
TEST_TOMOS="${TEST_TOMOS:-TS_5_4,TS_69_2}"
SSL_V1_CLASS="${SSL_V1_CLASS:-4}"
DIAMETERS="1:130,2:180,3:260,4:300,5:120,6:270"
NAMES="1:beta-amylase,2:beta-galactosidase,3:thyroglobulin,4:ribosome,5:apo-ferritin,6:virus-like-particle"
mkdir -p "$OUT"
log() { echo "[$(date '+%F %T')] $*"; }

if [[ "${TINY:-0}" == 1 ]]; then
    common=(--tiny)
else
    common=(--tomo_dir "$CZII/tomograms" --star "$CZII/coords/all_objects.star"
            --class_diameters "$DIAMETERS" --class_names "$NAMES"
            --epochs "$EPOCHS" --batch_size "$BATCH_SIZE" --ssl_v1_class "$SSL_V1_CLASS")
fi
ckpt_args() {  # SSL checkpoints given or produced by the fine-tuning step
    local a=()
    local v2="${SSL_V2_CKPT:-$(find "$OUT/finetune/pretrain_ssl_v2" -name '*.ckpt' 2>/dev/null | sort | tail -1)}"
    local v1="${SSL_V1_CKPT:-$(find "$OUT/finetune/pretrain_ssl_v1" -name '*.ckpt' 2>/dev/null | sort | tail -1)}"
    [[ -n "$v2" ]] && a+=(--ssl_v2 "$v2")
    [[ -n "$v1" ]] && a+=(--ssl_v1 "$v1")
    echo "${a[@]}"
}

# 1. Pretraining + fine-tuning + inference + scoring. SSL is pretrained on the training
#    tomograms only, so the test tomograms stay unseen.
if [[ -f "$OUT/finetune/report.json" ]]; then
    log "fine-tuning benchmark already done, skipping"
else
    log "fine-tuning benchmark -> $OUT/finetune (log: $OUT/finetune.log)"
    extra=()
    [[ "${TINY:-0}" == 1 ]] || extra=(--test_tomos "$TEST_TOMOS" --seeds "$SEEDS" --n_train "$N_TRAIN" --arms "$ARMS"
                                    --ssl_v1_epochs "$SSL_V1_EPOCHS" --ssl_v2_epochs "$SSL_V2_EPOCHS")
    # shellcheck disable=SC2046
    python benchmarks/ssl_finetune.py "${common[@]}" "${extra[@]}" $(ckpt_args) \
        --out "$OUT/finetune" > "$OUT/finetune.log" 2>&1
fi

# 2. Dense-embedding probe, reusing the SSL checkpoints from step 1.
if [[ -f "$OUT/probe/report.json" ]]; then
    log "probe already done, skipping"
else
    log "embedding probe -> $OUT/probe (log: $OUT/probe.log)"
    # shellcheck disable=SC2046
    python benchmarks/ssl_probe.py "${common[@]}" $(ckpt_args) --out "$OUT/probe" > "$OUT/probe.log" 2>&1
fi

# 3. Log the results.
if [[ "${TINY:-0}" != 1 ]]; then
    {
        printf '\n## SSL benchmarks on CZII (%s)\n\n' "$(date '+%F %H:%M')"
        printf -- '- output: `%s`; epochs %s (SSL v1 %s, v2 %s), seeds %s, n_train %s, arms %s, test %s\n\n' \
            "$OUT" "$EPOCHS" "$SSL_V1_EPOCHS" "$SSL_V2_EPOCHS" "$SEEDS" "$N_TRAIN" "$ARMS" "$TEST_TOMOS"
        printf '### Fine-tuning\n\n'; cat "$OUT/finetune/report.md" 2>/dev/null || echo "(no report.md)"
        printf '\n### Embedding probe\n\n'; cat "$OUT/probe/report.md" 2>/dev/null || echo "(no report.md)"
    } >> experiments/czii_v2_results.md
    log "results appended to experiments/czii_v2_results.md"
fi
log "done: $OUT"
