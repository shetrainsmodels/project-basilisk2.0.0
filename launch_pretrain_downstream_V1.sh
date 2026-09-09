#!/bin/bash

set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"   
mkdir -p logs                              

# Usage:  ./launch_pretrain_downstream_V1.sh <DATASET> [LAM ...]
#   DATASET = OPP (folds 1-4), PAM (folds 1-8), RW (folds 1-15) or RD (folds 1-17)
#   LAM     = JEPA-loss weights to launch; default list below.
# Examples: ./launch_pretrain_downstream_V1.sh OPP 0.1 0.3
#           ./launch_pretrain_downstream_V1.sh PAM 0 0.1
if [ -z "${1:-}" ]; then
    echo "Usage: $0 <DATASET: OPP|PAM|RW|RD> [LAM ...]"
    exit 1
fi
DATASET=$1
shift
LAMS=${@:-0 0.1 0.3 0.5 1}

case "$DATASET" in
    OPP) FOLDS="1 2 3 4" ;;
    PAM) FOLDS="1 2 3 4 5 6 7 8" ;;
    RW)  FOLDS="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15" ;;
    RD)  FOLDS="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17" ;;
    *)   echo "Unknown dataset: $DATASET (use OPP, PAM, RW or RD)"; exit 1 ;;
esac

for LAM in $LAMS; do
    for FOLD in $FOLDS; do
        echo "Submitting pretraining for ${DATASET} lam ${LAM} fold ${FOLD}"

        PRE_ID=$(sbatch --parsable --job-name=pre_${DATASET}_lam${LAM}_f${FOLD} pretrain_basilisk.slurm "$DATASET" "$FOLD" "$LAM")
        echo "Pretraining job for ${DATASET} lam ${LAM} fold ${FOLD}: ${PRE_ID}"
        echo "Submitting downstream dependent on ${PRE_ID}"

        DOWN_ID=$(sbatch --parsable --job-name=down_${DATASET}_lam${LAM}_f${FOLD} --dependency=afterok:${PRE_ID} train_downstream_basilisk.slurm "$DATASET" "$FOLD" "$LAM")
        echo "Downstream job ID for ${DATASET} lam ${LAM} fold ${FOLD}: ${DOWN_ID}"
    done
done
