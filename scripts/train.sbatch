#!/bin/bash
# LeWM training on Unity: one array task per square mode (0 baseline, 1 episode, 2 frame).
#
#   mkdir -p runs/slurm && sbatch scripts/train.sbatch                  # all three, seed 0
#   SEED=1 sbatch scripts/train.sbatch                                  # another seed
#   sbatch --array=1 scripts/train.sbatch                               # only the episode arm
#   sbatch --array=0 --time=0:20:00 scripts/train.sbatch --epochs 1 --max-steps-per-epoch 300 --run-name profile
#   PREFIX=hjepa_ sbatch --job-name=hjepa --time=12:00:00 scripts/train.sbatch --model hjepa   # runs/hjepa_<mode>_s0
#   PREFIX=idm_ sbatch --job-name=idm scripts/train.sbatch --idm-weight 100                    # runs/idm_<mode>_s0
#   PREFIX=hjepa_idm_ sbatch --job-name=hjepa-idm --time=12:00:00 scripts/train.sbatch --model hjepa --idm-weight 100 --idm-weight2 50
#
# Extra arguments are passed through to scripts/train.py. Jobs on gpu-preempt can be
# preempted; --requeue restarts them and train.py resumes from runs/<run>/last.pt.
#
#SBATCH --job-name=lewm
#SBATCH --partition=gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --constraint=2080ti
#SBATCH --cpus-per-task=6
#SBATCH --mem=8G
#SBATCH --time=8:00:00
#SBATCH --array=0-2
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=runs/slurm/%x-%A_%a.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

MODES=(baseline episode frame)
MODE=${MODES[$SLURM_ARRAY_TASK_ID]}
SEED=${SEED:-0}

export PATH="$HOME/.local/bin:$PATH" PYTHONUNBUFFERED=1
export UV_CACHE_DIR="$SLURM_SUBMIT_DIR/.uv-cache" UV_LINK_MODE=copy

echo "job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID on $(hostname): mode=$MODE seed=$SEED restart=${SLURM_RESTART_COUNT:-0}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# GPU memory as nvidia-smi sees it (includes the CUDA context, unlike torch's counters).
nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader -l 120 > "runs/slurm/gpu-$SLURM_JOB_ID.csv" &
SMI_PID=$!
trap 'kill $SMI_PID 2>/dev/null || true' EXIT

uv run --frozen python scripts/train.py \
    --square-mode "$MODE" --seed "$SEED" --run-name "${PREFIX:-}${MODE}_s${SEED}" \
    --num-workers $((SLURM_CPUS_PER_TASK - 1)) "$@"
