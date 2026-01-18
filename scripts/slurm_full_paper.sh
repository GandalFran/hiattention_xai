#!/bin/bash
#SBATCH --job-name=hiattention_paper
#SBATCH --output=logs/paper_run_%j.log
#SBATCH --error=logs/paper_run_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00

# Complete Pipeline for Paper Results

echo "============================================================"
echo "HiAttention-XAI Paper Experiments"
echo "============================================================"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Environment
export PATH="$HOME/miniconda3/bin:$PATH"
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate hiattention

cd ~/hiattention_xai
mkdir -p logs checkpoints results

# 1. Generate Dataset
echo ""
echo "============================================================"
echo "Step 1: Generating Dataset"
echo "============================================================"
# Only generate if not exists to save time, or force it? 
# Force it to be sure we use the latest logic.
python scripts/generate_full_dataset.py

# 2. Train Baseline
echo ""
echo "============================================================"
echo "Step 2: Training Baseline (TF-IDF)"
echo "============================================================"
python scripts/train_baseline.py

# 3. Train SOTA Proxy
echo ""
echo "============================================================"
echo "Step 3: Training SOTA Proxy (Transformer Only)"
echo "============================================================"
python scripts/train_sota_proxy.py

# 4. Train Proposed Model
echo ""
echo "============================================================"
echo "Step 4: Training HiAttention-XAI (Full Model)"
echo "============================================================"
python scripts/full_train_paper.py

echo ""
echo "============================================================"
echo "Experiments Complete!"
echo "Check results/ for .json metrics."
echo "============================================================"
