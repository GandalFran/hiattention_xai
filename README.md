# KG-HiAttention

**Synergizing AI-based Knowledge Graphs and Deep Learning for Explainable Software Vulnerability Analysis**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/pytorch-2.1+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A neuro-symbolic framework for software vulnerability analysis that combines:

- **Semantic Encoding**: Pre-trained CodeT5 transformer for token-level code understanding
- **Structural Encoding**: Graph Attention Networks (GAT) over a lightweight CPG-based knowledge graph (CFG/DFG)
- **Expert Knowledge**: Static analysis features for vulnerability patterns
- **Neuro-Symbolic Fusion**: Multi-modal integration for vulnerability prediction with explainability
- **Graph-grounded XAI**: Developer-facing explanations via attention attribution and SHAP on program graphs

## 🎯 Key Features

- **AI-based Knowledge Graphs**: CPG-inspired lightweight program graphs with typed CFG/DFG relations
- **Neuro-Symbolic Integration**: Combines neural representations (CodeT5) with symbolic structure (GAT over program graphs)
- **Multi-modal Fusion**: Semantic embeddings + structural graph context + expert static features
- **Explainability**: Graph-grounded explanations with faithfulness and stability proxies
- **Real-world Evaluation**: Tested on BigVul dataset (C/C++ vulnerabilities from Linux Kernel, Chrome, FFmpeg)
- **HPC Ready**: Optimized for single NVIDIA H100 80GB GPU

## 📊 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  INPUT: Source Code Function                │
└────────────────────────┬────────────────────────────────────┘
                         │
        ┌────────────────┴────────────────┐
        │                                 │
        ▼                                 ▼
   Token Sequence                 Lightweight CPG (KG)
   (CodeT5 tokenizer)            (CFG/DFG relations)
        │                                 │
        └───────────────┬─────────────────┘
                        │
        ┌───────────────▼───────────────────┐
        │      KG-HiAttention Model          │
        │                                    │
        │  Level 1: Input Representation    │
        │  Level 2: Semantic (CodeT5)       │
        │  Level 3: Structural (GAT)        │
        │  Level 4: Expert Knowledge        │
        │  Level 5: Neuro-Symbolic Fusion   │
        └───────────────┬───────────────────┘
                        │
        ┌───────────────▼───────────────────┐
        │      XAI Module                   │
        │  - Graph Attention Attribution    │
        │  - SHAP-style Explanations        │
        │  - Faithfulness/Stability Proxies │
        └───────────────┬───────────────────┘
                        │
                        ▼
      Vulnerability Score + Explanation
```

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/Gandalfran/hiattention_xai.git
cd hiattention_xai

# Create environment
conda create -n kg_hiattention python=3.11 -y
conda activate kg_hiattention

# Install dependencies
pip install -r requirements.txt
```

### Dataset Preparation

```bash
# Download BigVul dataset
bash scripts/download_datasets.sh

# Preprocess and build program graphs
python scripts/preprocess_data.py --dataset bigvul --output_dir datasets/processed
```

### Training

```bash
# Train KG-HiAttention model
python scripts/full_train_paper.py --config config/training_config.yaml --gpu 0

# Quick training for testing
python scripts/quick_train.py --epochs 5
```

### Generate Explanations

```bash
# Generate graph-grounded explanations
python scripts/visualize_graph_xai.py --checkpoint checkpoints/best.pt --output_dir results/explanations
```

## 📁 Project Structure

```
hiattention_xai/
├── config/                  # Training configurations
├── data/                    # CPG/KG building, preprocessing
│   ├── code_parser.py      # C/C++ code parsing
│   ├── graph_builder.py    # Lightweight CPG construction
│   └── preprocessor.py     # Dataset preprocessing
├── models/                  # Neuro-symbolic models
│   ├── semantic_encoder.py # CodeT5 encoding
│   ├── graph_encoder.py    # GAT for program graphs
│   ├── expert_encoder.py   # Static feature projection
│   └── kg_hiattention.py   # Complete fusion model
├── explainability/          # XAI components
│   ├── attribution.py      # Graph-based attribution
│   ├── shap_explainer.py   # SHAP integration
│   └── faithfulness.py     # Faithfulness/stability metrics
├── training/                # Training infrastructure
│   ├── trainer.py
│   └── metrics.py
├── evaluation/              # Evaluation framework
│   └── evaluator.py
└── utils/                   # Utilities

scripts/                     # Experiment scripts
├── download_datasets.sh     # Dataset acquisition
├── preprocess_data.py       # Data preprocessing
├── full_train_paper.py      # Main training script
├── quick_train.py           # Quick training for testing
└── visualize_graph_xai.py   # XAI visualization
```

## 🖥️ HPC Usage

### Setup Environment

```bash
# Setup conda environment on HPC
bash scripts/setup_hpc_env.sh
```

### Download BigVul Dataset

```bash
# Download and extract BigVul dataset
bash scripts/download_datasets.sh ./datasets
```

### Run Training on HPC

```bash
# Run training with SLURM
sbatch scripts/slurm_full_paper.sh

# Or run directly
bash scripts/run_training.sh
```

## 📈 Results on BigVul Dataset

Performance comparison on the BigVul test set (C/C++ vulnerabilities):

| Model Type | Method | AUC-ROC | Recall | Explainability |
|------------|--------|---------|--------|----------------|
| Traditional ML | Hybrid Ensemble | 0.7785 | 0.78 | Feature Importance |
| Deep Learning | CodeT5-Base | 0.7372 | 0.71 | Attention Weights |
| Deep Learning | KG-XAI (Single Fusion) | 0.7601 | 0.75 | Multi-Modal |
| Deep Learning | **KG-HiAttention (Ensemble)** | **0.7859** | **0.79** | **Multi-Modal** |

**Key Insights:**
- **KG-HiAttention (Ensemble) achieves AUC-ROC of 0.7859**, surpassing the strong Hybrid Ensemble baseline (0.7785)
- The neuro-symbolic fusion of semantic (CodeT5) and structural (CPG) features, when stabilized via ensemble learning and focal loss, provides a superior decision boundary
- The incremental improvement from single fusion (0.7601) to ensemble (0.7859) highlights the importance of mitigating variance in small, imbalanced datasets
- Graph-grounded explanations are complemented with faithfulness (deletion/insertion AUC: 0.847/0.823) and stability (mean consistency: 0.893) proxies

## 🔬 Reproducing Paper Results

To reproduce the results from the paper:

```bash
# 1. Preprocess BigVul dataset
python scripts/preprocess_data.py --dataset bigvul

# 2. Train baselines
python scripts/train_baseline.py --model codet5 --output_dir results/codet5
python scripts/train_baseline.py --model hybrid --output_dir results/hybrid

# 3. Train KG-HiAttention
python scripts/full_train_paper.py --output_dir results/kg_hiattention

# 4. Evaluate explainability
python scripts/visualize_graph_xai.py --checkpoint results/kg_hiattention/best.pt
```

## 📜 License

MIT License - see [LICENSE](LICENSE) for details.
