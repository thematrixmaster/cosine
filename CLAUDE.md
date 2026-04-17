# CoSiNE: Conditional Simulation of Evolutionary Sequences

A framework for training neural continuous-time Markov chain (CTMC) models of protein evolution using PyTorch Lightning and Hydra configuration management.

## Repository Structure

```
cosine/
  configs/            # Hydra configuration files
    train.yaml        # Main training configuration
    experiment/       # Experiment-specific configs
      train_ctmc_model_on_dasm.yaml  # CTMC training on DASM
    data/             # Data module configurations
      dataset/
        dasm-tr.yaml  # DASM training data
        dasm-val.yaml # DASM validation data
        ctmc.yaml     # CTMC dataset wrapper
    model/            # Model configurations
      ctmc.yaml       # CTMC model config
    net/              # Network architecture configs
      ctmc.yaml       # Neural CTMC network
    trainer/          # Lightning trainer configs
    callbacks/        # Training callbacks
  cosine/             # Main Python package
    models/           # Model implementations
      nets/
        ctmc.py       # Neural CTMC rate matrix parameterization
        encoder.py    # ESM-2 encoder wrapper
        esm2.py       # ESM-2 with flash attention
      modules/
        ctmc_module.py  # PyTorch Lightning training module
    data/             # Data modules and datasets
      datasets/
        ctmc.py       # CTMC dataset wrapper
        dasm.py       # DASM dataset loader
    utils/            # Utility functions
    metrics/          # Custom metrics
  evo/                # Submodule for biology utilities
  experiments/        # Training scripts
    train_model.py    # Main Hydra-based training
    eval_model.py     # Model evaluation
  scripts/            # Sampling and guidance scripts
    guidance/
      cosine.py       # CoSiNE guided sampling (TAG algorithm)
      genetic.py      # Genetic algorithm baseline
      mlm.py          # MLM baseline
      random_baseline.py  # Random baseline
    sampling/         # CTMC sampling utilities
      oracle_guided_evolution.py
      sample_from_ctmc_model.py
    structure/
      fold_with_abodybuilder3.py  # Antibody structure prediction
  notebooks/          # Jupyter notebooks for analysis
    eval_ctmc_model.ipynb  # Main evaluation (paper results)
    eval_ppl.ipynb         # Perplexity comparison
  docs/               # Documentation
    ctmc_guidance.md        # Oracle-guided sampling docs
    tag_implementation.md   # TAG algorithm details
  data/               # Dataset storage (not committed)
  checkpoints/        # Model checkpoints (not committed)
  train.sh            # SLURM job submission script
```

## Quick Start

### 1. Clone and Setup Environment

```bash
# Clone the repository
git clone <repository-url>
cd cosine

# Pull the evo submodule
git submodule update --init

# Create and activate virtual environment using uv
uv venv --python 3.10
source .venv/bin/activate

# Install dependencies
uv sync
```

### 2. Verify Installation

```bash
# Check that the package imports correctly
python -c "import cosine; print('Installation successful')"
```

## Training Models

### Main Training Framework

The repository uses PyTorch Lightning + Hydra for structured training:

```bash
# Basic training with default config
uv run python experiments/train_model.py experiment=train_ctmc_model_on_dasm

# Override specific parameters
uv run python experiments/train_model.py experiment=train_ctmc_model_on_dasm \
    trainer.max_epochs=50 \
    model.optimizer.lr=0.001

# Train with different settings
uv run python experiments/train_model.py experiment=train_ctmc_model_on_dasm \
    trainer.max_epochs=100 \
    data.batch_size=32 \
    model.reversible=true
```

### SLURM Cluster Training

For training on SLURM clusters:

```bash
# Submit job with default config
sbatch train.sh experiments/train_model.py experiment=train_ctmc_model_on_dasm

# Submit with parameter overrides
sbatch train.sh experiments/train_model.py experiment=train_ctmc_model_on_dasm trainer.max_epochs=200
```

## Configuration System

### Understanding Hydra Configs

The configuration system uses Hydra with a hierarchical structure:

- **Base config**: `configs/train.yaml` - sets default components
- **Component configs**: Individual modules (data, model, trainer, etc.)
- **Experiment configs**: `configs/experiment/` - complete experiment setups

### Key Configuration Files

- `configs/experiment/train_ctmc_model_on_dasm.yaml` - CTMC model training on DASM
- `configs/data/plmr.yaml` - Data loading configuration
- `configs/data/dataset/dasm-tr.yaml` - DASM training dataset
- `configs/data/dataset/dasm-val.yaml` - DASM validation dataset
- `configs/model/ctmc.yaml` - CTMC model configuration
- `configs/net/ctmc.yaml` - Neural CTMC network architecture

### Custom Experiments

Create new experiment configs in `configs/experiment/`:

```yaml
# @package _global_
defaults:
  - override /data: plmr
  - override /model: ctmc
  - override /trainer: gpu

tags: ["my_ctmc_experiment"]
seed: 42

trainer:
  max_epochs: 100

model:
  reversible: true
  optimizer:
    lr: 0.0003
```

## CoSiNE Model

### Neural CTMC Architecture

The CoSiNE model parameterizes position-specific rate matrices using:

- **ESM-2 Encoder**: Pre-trained protein language model for sequence embeddings
- **Rate Matrix Output Head**: Neural network that produces reversible rate matrices via Pande transformation
- **Training**: Maximum likelihood estimation on evolutionary sequence pairs

Configuration: `configs/model/ctmc.yaml` and `configs/net/ctmc.yaml`

### Key Model Features

- Context-dependent rate matrix parameterization
- Reversible or non-reversible rate matrices (configurable)
- ESM-2 integration for rich sequence representations

## Data Management

### DASM Dataset

The primary dataset is **DASM** (Deep Antibody Sequence Modeling):

- Store datasets in `./data/dasm/` directory
- Training data config: `configs/data/dataset/dasm-tr.yaml`
- Validation data config: `configs/data/dataset/dasm-val.yaml`
- See `data/README.md` for download instructions and data format

### Supported Data Formats

- Evolutionary sequence pairs (ancestor-descendant)
- FASTA sequences with phylogenetic distances
- Custom dataset formats via data modules

## Guided Sampling with CoSiNE

### Oracle-Guided Evolution (TAG Algorithm)

The **CoSiNE method** uses Test-time Augmented Guidance for optimizing biological properties:

```bash
# Run CoSiNE guided sampling
python scripts/guidance/cosine.py \
    --checkpoint checkpoints/cosine_model.ckpt \
    --oracle covid \
    --num_sequences 100 \
    --guidance_scale 2.0 \
    --mutation_ceiling 10
```

See `docs/ctmc_guidance.md` and `docs/tag_implementation.md` for detailed documentation.

### Baseline Comparisons

```bash
# Genetic algorithm baseline
python scripts/guidance/genetic.py --oracle covid

# MLM baseline
python scripts/guidance/mlm.py --oracle covid

# Random baseline
python scripts/guidance/random_baseline.py --oracle covid
```

## Development Workflow

### Code Structure

- Follow existing patterns in `cosine/` package
- Use Lightning modules for models (`cosine/models/modules/`)
- Implement datasets in `cosine/data/datasets/`
- Add utilities to `cosine/utils/`

### Adding New Components

1. **New Oracle**: Add to `evo/evo/oracles/` for guided sampling
2. **New Dataset**: Implement in `cosine/data/datasets/` and add config
3. **New Sampling Method**: Add to `scripts/guidance/` or `scripts/sampling/`

### Testing and Validation

**Test Organization Guidelines:**
- All tests MUST be placed in the `tests/` folder using proper pytest structure
- Use descriptive test file names starting with `test_`
- Follow pytest naming conventions for test functions (`test_*`)

```bash
# Run pytest test suite
uv run pytest tests/

# Run specific test file
uv run pytest tests/test_generative_metrics.py

# Run tests with verbose output
uv run pytest tests/ -v
```

## Analysis Notebooks

Jupyter notebooks in `notebooks/` provide:

- `eval_ctmc_model.ipynb` - Main CTMC model evaluation (paper results)
- `eval_ppl.ipynb` - Perplexity comparison with baselines
- `eval_consistency.ipynb` - Model consistency analysis
- `sample_dasm.ipynb` - DASM dataset sampling examples

## Common Commands

```bash
# List available configurations
uv run python experiments/train_model.py --help

# Validate configuration without training
uv run python experiments/train_model.py --cfg job

# Run with debugging
uv run python experiments/train_model.py debug=default

# Resume from checkpoint
uv run python experiments/train_model.py ckpt_path=/path/to/checkpoint.ckpt

# Multi-GPU training
uv run python experiments/train_model.py trainer=ddp trainer.devices=4
```

## Monitoring and Logging

- **Weights & Biases**: Configure in `configs/logger/wandb.yaml`
- **CSV Logging**: Built-in via `configs/logger/csv.yaml`
- **Checkpoints**: Saved automatically during training
- **Metrics**: Custom metrics in `cosine/metrics/`

## Tips for Development

1. **Start with Paper Results**: Run `eval_ctmc_model.ipynb` to understand model behavior
2. **Check Guided Sampling**: Explore `scripts/guidance/cosine.py` for TAG implementation
3. **Use Hydra**: Leverage configuration composition for reproducible experiments
4. **Monitor Training**: Set up logging early in development
5. **Test Sampling**: Use `scripts/sampling/sample_from_ctmc_model.py` for quick tests

## Troubleshooting

- **Import errors**: Ensure virtual environment is activated and dependencies installed
- **CUDA issues**: Check PyTorch installation and GPU availability
- **Configuration errors**: Use `--cfg job` to validate configs before training
- **Memory issues**: Reduce batch size in data configuration
- **Slow sampling**: Ensure rate matrices are properly normalized

## Key Implementation Files

- **Neural CTMC**: `cosine/models/nets/ctmc.py`
- **Training Module**: `cosine/models/modules/ctmc_module.py`
- **TAG Algorithm**: `scripts/guidance/cosine.py`
- **ESM Encoder**: `cosine/models/nets/encoder.py`

## IMPORTANT INSTRUCTIONS

- Only make minimal code edits required
- Focus on CTMC/CoSiNE model and DASM dataset
- Use existing configs in `configs/experiment/train_ctmc_model_on_dasm.yaml` as template
- For guided sampling, refer to `scripts/guidance/cosine.py` implementation
