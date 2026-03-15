#!/bin/bash
#SBATCH -A MLMI-GHDL2-SL2-GPU
#SBATCH --job-name=tabicl_stage1
#SBATCH --output=/home/ghdl2/rds/hpc-work/tabicl/logs/tabicl_stage1_%j.out
#SBATCH --error=/home/ghdl2/rds/hpc-work/tabicl/logs/tabicl_stage1_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH -p ampere
##SBATCH --no-requeue

# Enable module command
. /etc/profile.d/modules.sh

module purge
module load rhel8/default-amp
module load cuda/12.1

PROJECT_DIR="/home/ghdl2/rds/hpc-work/tabicl"
PYTHON_SCRIPT="train.run"

WANDB_DIR="${PROJECT_DIR}/wandb"
CHECKPOINT_DIR="${PROJECT_DIR}/stage1_checkpoints"

set -euo pipefail

mkdir -p ${PROJECT_DIR}/logs
mkdir -p "${WANDB_DIR}"
mkdir -p "${CHECKPOINT_DIR}"

cd "${PROJECT_DIR}"

# Activate python environment
source venv/bin/activate

echo "========================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "GPU info:"
nvidia-smi
echo "Working directory: $(pwd)"
echo "Starting at: $(date)"
echo "========================================"

torchrun --standalone --nproc_per_node=1 -m train.run \
    --wandb_log True \
    --wandb_project TabICL \
    --wandb_name Stage1 \
    --wandb_dir "${WANDB_DIR}" \
    --wandb_mode online \
    --device cuda \
    --dtype float32 \
    --np_seed 42 \
    --torch_seed 42 \
    --max_steps 2000 \
    --batch_size 512 \
    --micro_batch_size 1 \
    --lr 2e-5 \
    --scheduler polynomial_decay_warmup \
    --warmup_proportion 0 \
    --poly_decay_lr_end 5e-6 \
    --poly_decay_power 2.0 \
    --gradient_clipping 1.0 \
    --prior_type mix_scm \
    --prior_device cpu \
    --batch_size_per_gp 2 \
    --min_features 2 \
    --max_features 100 \
    --max_classes 10 \
    --min_seq_len 1000 \
    --max_seq_len 40000 \
    --log_seq_len True \
    --seq_len_per_gp True \
    --min_train_size 0.5 \
    --max_train_size 0.9 \
    --embed_dim 128 \
    --col_num_blocks 3 \
    --col_nhead 4 \
    --col_num_inds 128 \
    --row_num_blocks 3 \
    --row_nhead 8 \
    --row_num_cls 4 \
    --row_rope_base 100000 \
    --icl_num_blocks 12 \
    --icl_nhead 4 \
    --ff_factor 2 \
    --norm_first True \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --save_temp_every 5 \
    --save_perm_every 100

echo "========================================"
echo "Finished at: $(date)"
echo "========================================"