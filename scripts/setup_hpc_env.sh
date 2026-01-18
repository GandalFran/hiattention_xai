#!/bin/bash
# HPC Environment Setup Script for HiAttention-XAI
# Run this on the HPC cluster to setup the conda environment

set -e

echo "=========================================="
echo "HiAttention-XAI HPC Environment Setup"
echo "=========================================="

# Configuration
ENV_NAME="hiattention"
PYTHON_VERSION="3.11"
CUDA_VERSION="12.1"

# Check for conda
if ! command -v conda &> /dev/null; then
    echo "Error: conda not found. Please install Miniconda or Anaconda first."
    exit 1
fi

echo ""
echo "Step 1: Creating conda environment..."
conda create -n $ENV_NAME python=$PYTHON_VERSION -y

echo ""
echo "Step 2: Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

echo ""
echo "Step 3: Installing PyTorch with CUDA $CUDA_VERSION..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo ""
echo "Step 4: Installing PyTorch Geometric..."
pip install torch-geometric
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

echo ""
echo "Step 5: Installing other dependencies..."
pip install transformers>=4.30.0 tokenizers>=0.13.0
pip install shap>=0.42.0 captum>=0.6.0
pip install numpy>=1.24.0 pandas>=2.0.0 h5py>=3.8.0 networkx>=3.0
pip install matplotlib>=3.7.0 seaborn>=0.12.0
pip install scikit-learn>=1.2.0
pip install pyyaml>=6.0 wandb>=0.15.0
pip install tqdm>=4.65.0 rich>=13.0.0
pip install python-dotenv>=1.0.0
pip install pytest>=7.3.0

echo ""
echo "Step 6: Verifying installation..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"
python -c "import torch_geometric; print(f'PyG: {torch_geometric.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"

echo ""
echo "=========================================="
echo "Environment setup complete!"
echo "To activate: conda activate $ENV_NAME"
echo "=========================================="
