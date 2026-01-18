#!/bin/bash
# HiAttention-XAI Training Runner for HPC
# Submits distributed training job using srun

set -e

# Configuration
PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/datasets/processed}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$PROJECT_DIR/checkpoints}"
LOG_DIR="${LOG_DIR:-$PROJECT_DIR/logs}"
CONFIG="${CONFIG:-$PROJECT_DIR/hiattention_xai/config/training_config.yaml}"

# GPU Configuration (6 of 8 H100 GPUs)
NUM_GPUS=6
BATCH_SIZE=128
EPOCHS=50

echo "=========================================="
echo "HiAttention-XAI Training Launcher"
echo "=========================================="
echo "Project:     $PROJECT_DIR"
echo "Data:        $DATA_DIR"
echo "Checkpoints: $CHECKPOINT_DIR"
echo "GPUs:        $NUM_GPUS"
echo "=========================================="

# Create directories
mkdir -p "$CHECKPOINT_DIR" "$LOG_DIR"

# Activate environment
source ~/.bashrc
conda activate hiattention || {
    echo "Error: Environment 'hiattention' not found. Run setup_hpc_env.sh first."
    exit 1
}

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export WORLD_SIZE=$NUM_GPUS

# Timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_NAME="hiattention_xai_$TIMESTAMP"

echo ""
echo "Starting distributed training..."
echo "Experiment: $EXPERIMENT_NAME"
echo ""

# Run with srun for HPC
srun --user=hpc-login --gres=gpu:$NUM_GPUS --cpus-per-task=48 --mem=256G \
    python -m torch.distributed.launch \
    --nproc_per_node=$NUM_GPUS \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    "$PROJECT_DIR/train.py" \
    --config "$CONFIG" \
    --data_dir "$DATA_DIR" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --distributed \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --wandb_project hiattention-xai \
    --experiment_name "$EXPERIMENT_NAME" \
    2>&1 | tee "$LOG_DIR/training_$TIMESTAMP.log"

echo ""
echo "=========================================="
echo "Training complete!"
echo "Logs:        $LOG_DIR/training_$TIMESTAMP.log"
echo "Checkpoints: $CHECKPOINT_DIR"
echo "=========================================="
