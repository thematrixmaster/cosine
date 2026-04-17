# CoSiNE: Conditional Simulation of Evolutionary Sequences with Neural CTMC Models

Official implementation of **CoSiNE** (Conditional Simulation of Evolutionary Sequences), a framework for learning and sampling from neural continuous-time Markov chain (CTMC) models of protein evolution.

## Overview

CoSiNE learns sequence-conditional rate matrices that capture evolutionary dynamics from antibody sequence data. The model combines:

- **Neural CTMC**: Context-dependent parameterization of substitution rate matrices using ESM-2 embeddings
- **Test-time Augmented Guidance (TAG)**: Oracle-guided sampling for optimizing biological properties

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/cosine.git
cd cosine

# Pull the evo submodule
git submodule update --init
```

### 2. Set Up Environment

We recommend using [uv](https://github.com/astral-sh/uv) for fast dependency management:

```bash
# Create and activate virtual environment
uv venv --python 3.10
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv sync
```

### 3. Verify Installation

```bash
python -c "import cosine; print('CoSiNE installation successful!')"
```

## Data

### DASM Dataset

The primary dataset used for training is **DASM** (Deep Antibody Sequence Modeling dataset).

**TODO: Add download link**

```bash
# Download and extract DASM data
# mkdir -p data/dasm
# wget [TODO: ADD DOWNLOAD LINK] -O data/dasm.tar.gz
# tar -xzf data/dasm.tar.gz -C data/
```

See [`data/README.md`](data/README.md) for detailed data format documentation.

## Model Checkpoints

Pre-trained model checkpoints are available for reproducing paper results.

**TODO: Add download link**

```bash
# Download pre-trained checkpoints
# mkdir -p checkpoints
# wget [TODO: ADD DOWNLOAD LINK] -O checkpoints/cosine_model.ckpt
```

See [`checkpoints/README.md`](checkpoints/README.md) for checkpoint details and usage.

## Quick Start

### Training from Scratch

Train the CoSiNE model on DASM dataset:

```bash
# Basic training with default settings
uv run python experiments/train_model.py experiment=train_ctmc_model_on_dasm

# Custom training parameters
uv run python experiments/train_model.py \
    experiment=train_ctmc_model_on_dasm \
    trainer.max_epochs=100 \
    model.optimizer.lr=0.0003
```

### SLURM Cluster Training

For distributed training on SLURM clusters:

```bash
sbatch train.sh experiments/train_model.py experiment=train_ctmc_model_on_dasm
```

## Reproducing Paper Results

### 1. Evaluate CoSiNE Model

Run the main evaluation notebook to assess model performance:

```bash
jupyter notebook notebooks/eval_ctmc_model.ipynb
```

This notebook includes:
- Likelihood evaluation on held-out sequences
- Trajectory sampling visualization
- Rate matrix analysis

### 2. Perplexity Comparison

Compare CoSiNE against baselines (DASM + Thrifty):

```bash
jupyter notebook notebooks/eval_ppl.ipynb
```

### 3. Oracle-Guided Sampling (CoSiNE Method)

The **CoSiNE guided sampling algorithm** uses Test-time Augmented Guidance (TAG) for property optimization:

```bash
# Run CoSiNE guided sampling
python scripts/guidance/cosine.py \
    --checkpoint checkpoints/cosine_model.ckpt \
    --oracle covid \
    --num_sequences 100 \
    --guidance_scale 2.0 \
    --mutation_ceiling 10 \
    --output results/guided_sequences.csv
```

**Key Parameters:**
- `--oracle`: Oracle function for guidance (e.g., `covid`, `humanness`)
- `--guidance_scale`: Strength of guidance (higher = stronger optimization)
- `--mutation_ceiling`: Maximum mutations allowed from seed sequence
- `--region_mask`: Restrict mutations to specific regions (e.g., CDR loops)

See [`scripts/guidance/README.md`](scripts/guidance/) and [`docs/ctmc_guidance.md`](docs/ctmc_guidance.md) for detailed documentation.

### 4. Baseline Comparisons

Compare CoSiNE against other optimization methods:

```bash
# Genetic algorithm baseline
python scripts/guidance/genetic.py --oracle covid --num_sequences 100

# MLM (masked language model) baseline
python scripts/guidance/mlm.py --oracle covid --num_sequences 100

# Random baseline
python scripts/guidance/random_baseline.py --oracle covid --num_sequences 100
```

## Repository Structure

```
cosine/
├── cosine/                      # Main Python package
│   ├── models/
│   │   ├── nets/
│   │   │   ├── ctmc.py         # Neural CTMC implementation
│   │   │   ├── encoder.py      # ESM-2 encoder wrapper
│   │   │   └── esm2.py         # ESM-2 flash attention
│   │   └── modules/
│   │       ├── ctmc_module.py  # PyTorch Lightning training module
│   │       └── plmr_module.py  # Base module
│   ├── data/
│   │   ├── datasets/
│   │   │   ├── ctmc.py         # CTMC dataset wrapper
│   │   │   └── dasm.py         # DASM dataset
│   │   └── datamodule.py       # Data loading
│   ├── utils/                   # Utility functions
│   └── metrics/                 # Evaluation metrics
├── configs/                     # Hydra configuration files
│   ├── experiment/
│   │   └── train_ctmc_model_on_dasm.yaml
│   ├── model/ctmc.yaml
│   ├── net/ctmc.yaml
│   └── data/dataset/
│       ├── dasm-tr.yaml
│       └── dasm-val.yaml
├── experiments/                 # Training scripts
│   ├── train_model.py          # Main training script
│   └── eval_model.py           # Evaluation script
├── scripts/
│   ├── guidance/               # Guided sampling methods
│   │   ├── cosine.py          # CoSiNE method (TAG)
│   │   ├── genetic.py         # Genetic algorithm baseline
│   │   ├── mlm.py             # MLM baseline
│   │   └── random_baseline.py # Random baseline
│   └── sampling/               # CTMC sampling utilities
│       ├── oracle_guided_evolution.py
│       └── sample_from_ctmc_model.py
├── notebooks/                   # Analysis notebooks
│   ├── eval_ctmc_model.ipynb   # Main evaluation (paper results)
│   └── eval_ppl.ipynb          # Perplexity comparison
├── docs/                        # Documentation
│   ├── ctmc_guidance.md        # Guided sampling docs
│   └── tag_implementation.md   # TAG algorithm details
├── evo/                         # Submodule for biology utilities
├── data/                        # Dataset storage
└── checkpoints/                 # Model checkpoints
```

## Key Concepts

### Neural CTMC

The CoSiNE model parameterizes position-specific rate matrices Q(x, i) using neural networks:

- **Input**: ESM-2 embeddings of sequence context
- **Output**: Reversible rate matrices via Pande transformation
- **Training**: Maximum likelihood on observed evolutionary transitions

### Test-time Augmented Guidance (TAG)

Oracle-guided Gillespie sampling with augmented state space for property optimization. See `scripts/guidance/cosine.py` for implementation.

## Configuration System

CoSiNE uses [Hydra](https://hydra.cc/) for configuration management:

- **Base config**: `configs/train.yaml`
- **Experiment configs**: `configs/experiment/train_ctmc_model_on_dasm.yaml`
- **Override parameters**: Use command-line arguments

Example:
```bash
python experiments/train_model.py \
    experiment=train_ctmc_model_on_dasm \
    model.reversible=true \
    model.optimizer.lr=0.0001 \
    trainer.max_epochs=50
```

## Citation

If you use CoSiNE in your research, please cite:

```bibtex
@article{Lu2026ConditionallySN,
  title={Conditionally Site-Independent Neural Evolution of Antibody Sequences},
  author={Stephen Zhewen Lu and Aakarsh Vermani and Kohei Sanno and Jiarui Lu and IV FrederickA.Matsen and Milind Jagota and Yun S. Song},
  journal={ArXiv},
  year={2026},
  url={https://api.semanticscholar.org/CorpusID:285973749}
}
```

## License

TODO: Add license information

## Contact

For questions or issues, please open a GitHub issue or contact:
- Stephen Z. Lu (stephen.lu@berkeley.edu)

## Acknowledgments

This work builds on:
- [ESM](https://github.com/facebookresearch/esm) - Protein language models
- [CherryML](https://github.com/songlab-cal/CherryML) - Evolutionary pair data
- [PyTorch Lightning](https://www.pytorchlightning.ai/) - Training framework
- [Hydra](https://hydra.cc/) - Configuration management
