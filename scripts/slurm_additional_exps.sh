#!/bin/bash
#SBATCH --job-name=additional_exps
#SBATCH --output=logs/additional_exps_%j.log
#SBATCH --error=logs/additional_exps_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00

echo "Starting additional experiments"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-interactive}"


export PATH="$HOME/miniconda3/bin:$PATH"
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate hiattention

cd ~/hiattention_xai
mkdir -p logs outputs

echo ""
echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")')"


pip install scipy --quiet 2>/dev/null

export PYTHONUNBUFFERED=1

echo ""
echo "Running experiments 1, 3, 7 (4 and 6 are already done)"

python -u scripts/run_additional_experiments.py --exp 1 --gpu 0 2>&1
echo "--- EXP-1 done at $(date) ---"

python -u scripts/run_additional_experiments.py --exp 3 --gpu 0 2>&1
echo "--- EXP-3 done at $(date) ---"

python -u scripts/run_additional_experiments.py --exp 7 --gpu 0 2>&1
echo "--- EXP-7 done at $(date) ---"

echo ""
echo "Finished at $(date)"
echo "Output files:"
ls -la outputs/
