# HiAttention-XAI

**Hierarchical Attention-based Deep Learning for Context-Aware Software Defect Localization with Explainability**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A novel 5-level hierarchical deep learning architecture for software defect prediction that combines:

- **Level 2**: Local Context Encoder (CodeT5 + BiLSTM + Multi-Head Attention)
- **Level 3**: Function Dependency GNN (Graph Neural Networks for call/data-flow graphs)
- **Level 4**: Architectural Context Analyzer (Modularity, Coupling, Cohesion, Technical Debt)
- **Level 5**: Prediction & Fusion Head with built-in explainability

## 🎯 Key Features

- **Hierarchical Context**: Captures line, function, and module-level patterns
- **Graph-based Dependencies**: Models function calls and data flow with GNNs
- **Built-in Explainability**: Saliency maps, SHAP values, attention visualization
- **Fairness Analysis**: SPD, EOD, AOD metrics across protected attributes
- **Industrial Scale**: Designed for multi-million LOC repositories
- **HPC Ready**: Distributed training on 6+ NVIDIA H100 GPUs

## 📊 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  INPUT: Source Code                         │
└────────────────────────┬────────────────────────────────────┘
                         │
        ┌────────────────┴────────────────┐
        ▼                                 ▼
   Code Parsing                   Dependency Extraction
   (AST Analysis)                 (Call Graph, Data Flow)
        │                                 │
        └───────────────┬─────────────────┘
                        │
        ┌───────────────▼───────────────────┐
        │     HiAttention-XAI Model          │
        │                                    │
        │  Level 2: Local Context Encoder   │
        │  Level 3: Function Dependency GNN │
        │  Level 4: Architectural Context   │
        │  Level 5: Prediction + Fusion     │
        └───────────────┬───────────────────┘
                        │
        ┌───────────────▼───────────────────┐
        │      Explainability Module        │
        │  - Saliency Maps                  │
        │  - SHAP Values                    │
        │  - Attention Patterns             │
        └───────────────┬───────────────────┘
                        │
                        ▼
              Prediction + Explanation
```

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/GandalFran/hiattention-xai.git
cd hiattention-xai

# Create environment
conda create -n hiattention python=3.11 -y
conda activate hiattention

# Install dependencies
pip install -r requirements.txt
```

### Training

```bash
# Single GPU
python train.py --config hiattention_xai/config/training_config.yaml

# Distributed (6 GPUs)
torchrun --nproc_per_node=6 train.py --config hiattention_xai/config/training_config.yaml --distributed
```

### Evaluation

```bash
python evaluate.py --checkpoint checkpoints/best.pt --data_dir datasets/processed
```

### Generate Explanations

```bash
python explain.py --checkpoint checkpoints/best.pt --code_file sample.py --output_format markdown
```

## 📁 Project Structure

```
hiattention_xai/
├── config/             # Training configs, SLURM scripts
├── data/               # Code parsing, graph building, preprocessing
│   ├── code_parser.py
│   ├── graph_builder.py
│   ├── preprocessor.py
│   └── defect_seeder.py
├── models/             # 5-level hierarchical model
│   ├── local_context.py      # Level 2
│   ├── function_gnn.py       # Level 3
│   ├── architectural.py      # Level 4
│   ├── prediction_head.py    # Level 5
│   └── hiattention_xai.py    # Complete model
├── explainability/     # XAI components
│   ├── saliency.py
│   ├── shap_explainer.py
│   ├── attention_viz.py
│   └── report_generator.py
├── training/           # Training infrastructure
│   ├── trainer.py
│   ├── losses.py
│   └── metrics.py
├── evaluation/         # Evaluation framework
│   └── evaluator.py
└── utils/              # Utilities
    ├── logging_utils.py
    └── visualization.py

dataset.rar             # Compressed HDF5 datasets (extract to use)
                        # Contains: train.h5 (20K), val.h5 (5K), test.h5 (5K)

scripts/                # HPC and data scripts
tests/                  # Unit tests
train.py               # Training entry point
evaluate.py            # Evaluation entry point
explain.py             # Explanation generation
```

## 🖥️ HPC Usage

### Setup Environment

```bash
bash scripts/setup_hpc_env.sh
```

### Download Datasets

```bash
bash scripts/download_datasets.sh ./datasets
```

### Run Training on HPC

```bash
bash scripts/run_training.sh
```

Or submit via SLURM:

```bash
sbatch hiattention_xai/config/slurm_job.sh
```

## 📊 Dataset Information

The `dataset/` folder contains (compresed in a rar file) preprocessed HDF5 files ready for training and evaluation. The **"Context-Sensitive" validation protocol** generates controlled datasets with deterministic vulnerability patterns (SQL injection, buffer overflows, null dereferences, etc.) to rigorously validate the architecture's ability to capture specific defect structures.

### Dataset Files

| File | Samples | Description |
|------|---------|-------------|
| `train.h5` | 20,000 | Training dataset for model optimization |
| `val.h5` | 5,000 | Validation dataset for hyperparameter tuning |
| `test.h5` | 5,000 | Test dataset for final evaluation |

### Dataset Generation

To regenerate or modify the dataset:

```bash
# Generate synthetic context-sensitive dataset
python scripts/generate_full_dataset.py

# Or generate with custom parameters
python scripts/generate_synthetic.py --samples 25000 --max-length 256
```

## �📈 Expected Results

| Metric | HiAttention-XAI | PLEASE | LineVul |
|--------|-----------------|--------|---------|
| Recall@Top20% | **0.75** | 0.67 | 0.58 |
| F1 Score | **0.52** | 0.45 | 0.41 |
| AUC-ROC | **0.87** | 0.82 | 0.78 |

## 📜 License

MIT License - see [LICENSE](LICENSE) for details.
