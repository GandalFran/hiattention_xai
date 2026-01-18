#!/bin/bash
#SBATCH --job-name=hiattention_demo
#SBATCH --output=logs/demo_train_%j.log
#SBATCH --error=logs/demo_train_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00

# HiAttention-XAI Demo Training Job
# Single GPU, 5 epochs on synthetic data

echo "=========================================="
echo "HiAttention-XAI Demo Training"
echo "=========================================="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Environment
export PATH="$HOME/miniconda3/bin:$PATH"
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate hiattention

cd ~/hiattention_xai
mkdir -p logs checkpoints

echo ""
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

echo ""
echo "Starting training..."
python scripts/quick_train.py

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
