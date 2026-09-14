#!/bin/bash
#SBATCH --job-name=down_basilisk
#SBATCH --partition=contrib-gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32gb
#SBATCH --time=20:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err


set -euo pipefail

if [ -z "${1:-}" ] || [ -z "${2:-}" ] || [ -z "${3:-}" ]; then
	echo "Usage: sbatch train_downstream_basilisk.slurm <DATASET_NAME> <FOLD_ID> <LAM> [W_PER_CLASS]"
	echo "Example: sbatch train_downstream_basilisk.slurm OPP 1 0.1        # all labels"
	echo "Example: sbatch train_downstream_basilisk.slurm OPP 1 0.1 10     # 10 windows per class"
	exit 1
fi

JOB_START=$(date +%s)

echo "Job started on $(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi

source /home/kmercad/myenv/bin/activate
cd "$SLURM_SUBMIT_DIR"

python train_downstream_basilisk.py --dataset "$1" --fold "$2" --lam "$3" --w_per_class "${4:-all}"
RC=$?

JOB_END=$(date +%s)
ELAPSE=$((JOB_END-JOB_START))
echo "Total wall time:$((ELAPSE/60))min $((ELAPSE%60))s"
exit ${RC:-1}
