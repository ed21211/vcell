#!/usr/bin/env bash
# run_finetune_pilots.sh -- the six pilot runs: {single, dual} x LR {5e-7, 1e-6, 2e-6}.
# For each run: fine-tune on the balanced split -> embed every spot with the best checkpoint
# -> build image / ST / both datasets -> evaluate each arm (Ridge, fixed split, per-slide Pearson).
#
# Usage, from the vcell folder with the omiclip env active:
#     bash scripts/run_finetune_pilots.sh 3 2>&1 | tee outputs/ft_pilots.log
#   first argument = epochs per run (default 3)
#   MODES="single" LRS="1e-6" bash scripts/run_finetune_pilots.sh 3     # a subset
#   SKIP_SMOKE=1 bash scripts/run_finetune_pilots.sh 3                  # skip the 2-minute plumbing check
#
# It stops at the first error and can be started again: finished steps are skipped.
# Selection of the winning run is by validation Pearson of the three evaluation logs, not by
# the contrastive loss (see outputs/ft_logs/<tag>_eval_<arm>.log).

set -euo pipefail
cd "$(dirname "$0")/.."

EPOCHS="${1:-3}"
MODES="${MODES:-single dual}"
LRS="${LRS:-5e-7 1e-6 2e-6}"

META=outputs/omiclip_factor_baseline/omiclip_factor_metadata.csv
RESTRICT=features/hest_ccrcc_joined/hest_ccrcc_virchow2_factor_metadata.csv
SAMPLES=datasets/hest_metadata/ccrcc_samples.csv
SPLITS=datasets/splits/ccrcc_splits_v1.csv
ARMS=outputs/omiclip_ft/arms

mkdir -p outputs/omiclip_ft "$ARMS" models/omiclip_ft outputs/ft_logs

if [ "${SKIP_SMOKE:-0}" != "1" ]; then
  echo "== plumbing check (4 batches per mode)"
  for MODE in $MODES; do
    S="models/omiclip_ft/_smoke_${MODE}"
    rm -rf "$S"
    python scripts/finetune_omiclip_modes.py --mode "$MODE" --smoke --out-dir "$S" 2>&1 | tee "outputs/ft_logs/smoke_${MODE}_train.log"
    python scripts/extract_omiclip_ckpt.py --finetuned "$S/best.pt" --max-rows 256 --num-workers 2 \
      --output-h5 "$S/smoke_embeddings.h5" 2>&1 | tee "outputs/ft_logs/smoke_${MODE}_extract.log"
  done
fi

if [ ! -f models/omiclip_ft/frozen/done.json ]; then
  echo "== frozen reference (validation contrastive metrics of the pretrained model)"
  python scripts/finetune_omiclip_modes.py --mode frozen --out-dir models/omiclip_ft/frozen 2>&1 | tee outputs/ft_logs/frozen_train.log
fi

for MODE in $MODES; do
  for LR in $LRS; do
    TAG="${MODE}_lr${LR}"
    DIR="models/omiclip_ft/${TAG}"
    H5="outputs/omiclip_ft/${TAG}_embeddings.h5"
    echo "== ${TAG}"

    if [ ! -f "${DIR}/done.json" ]; then
      python scripts/finetune_omiclip_modes.py --mode "$MODE" --lr "$LR" --epochs "$EPOCHS" --out-dir "$DIR" \
        2>&1 | tee "outputs/ft_logs/${TAG}_train.log"
    fi

    if [ ! -f "$H5" ]; then
      python scripts/extract_omiclip_ckpt.py --finetuned "${DIR}/best.pt" --output-h5 "${H5}.tmp" \
        2>&1 | tee "outputs/ft_logs/${TAG}_extract.log"
      mv "${H5}.tmp" "$H5"
    fi

    if [ ! -f "${ARMS}/ft_${TAG}_both_factor_dataset.npz" ]; then
      python scripts/make_omiclip_arms.py --h5 "$H5" --image-h5 "$H5" --rebuild-image \
        --prefix "ft_${TAG}" --out-dir "$ARMS" 2>&1 | tee "outputs/ft_logs/${TAG}_arms.log"
    fi

    for ARM in image st both; do
      python scripts/evaluate_embeddings.py --name "ft_${TAG}_${ARM}" \
        --npz "${ARMS}/ft_${TAG}_${ARM}_factor_dataset.npz" --meta "$META" --restrict-to "$RESTRICT" \
        --sample-table "$SAMPLES" --split-csv "$SPLITS" --no-mlp --select within \
        2>&1 | tee "outputs/ft_logs/${TAG}_eval_${ARM}.log"
    done
  done
done

echo "== all runs finished"
