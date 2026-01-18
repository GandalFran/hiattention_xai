#!/bin/bash
#SBATCH --job-name=hiattention_xai
#SBATCH --output=logs/training_%j.out
#SBATCH --error=logs/training_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:6
#SBATCH --mem=256G
#SBATCH --time=72:00:00
#SBATCH --partition=gpu

# HiAttention-XAI Training Job Script for HPC
# Uses 6 of 8 NVIDIA H100 GPUs via srun

echo "=========================================="
echo "HiAttention-XAI Training Job"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: 6 x NVIDIA H100"
echo "Start time: $(date)"
echo "=========================================="

# Environment setup
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1

# Activate conda environment
source ~/.bashrc
conda activate hiattention

# Navigate to project directory
cd $SLURM_SUBMIT_DIR

# Set distributed training variables
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export WORLD_SIZE=6

echo "Master address: $MASTER_ADDR"
echo "Master port: $MASTER_PORT"

# Run training with srun
srun --user=hpc-login python -m torch.distributed.launch \
    --nproc_per_node=6 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train.py \
    --config hiattention_xai/config/training_config.yaml \
    --distributed \
    --wandb_project hiattention-xai \
    --experiment_name "full_training_$(date +%Y%m%d_%H%M%S)"

echo "=========================================="
echo "Training completed at: $(date)"
echo "=========================================="
