#!/bin/bash
#SBATCH -A MLMI-bbg25-SL2-GPU
#SBATCH -p ampere
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=250G
#SBATCH --time=12:00:00
#SBATCH --job-name=localized_tabicl
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# ── Environment ──────────────────────────────────────────────
source /rds/project/rds-xyBFuSj0hm0/MLMI6.L2026/miniconda3/bin/activate mlmi4

# ── Navigate to project directory ────────────────────────────
cd "$SLURM_SUBMIT_DIR"

echo "============================================"
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $SLURM_NODELIST"
echo "GPU:           $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Python:        $(python --version)"
echo "Working dir:   $(pwd)"
echo "Start time:    $(date)"
echo "============================================"

# ── Run experiments ──────────────────────────────────────────
python run_experiments.py

echo "============================================"
echo "End time:      $(date)"
echo "============================================"
