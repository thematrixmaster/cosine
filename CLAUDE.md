# PEINT: Protein Evolution INference Toolkit

A framework for training biological sequence evolution models using PyTorch Lightning and Hydra configuration management.

## Repository Structure

```
peint/
  configs/            # Hydra configuration files
    train.yaml        # Main training configuration
    experiment/       # Experiment-specific configs
    data/             # Data module configurations
    model/            # Model configurations
    net/              # Network architecture configs
    trainer/          # Lightning trainer configs
    callbacks/        # Training callbacks
  peint/              # Main Python package
    models/           # Model implementations
    data/             # Data modules and datasets
    utils/            # Utility functions
    metrics/          # Custom metrics
  evo/                # Submodule for biology utilities
  experiments/        # Standalone training scripts
  scripts/            # Validation and inference scripts
  notebooks/          # Jupyter notebooks for analysis
  data/               # Dataset storage (not committed)
  train.sh            # SLURM job submission script
```

## Quick Start

### 1. Clone and Setup Environment

```bash
# Clone the repository
git clone <repository-url>
cd peint

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
python -c "import peint; print('Installation successful')"
```

## Training Models

### Main Training Framework (Recommended)

The repository uses PyTorch Lightning + Hydra for structured training:

```bash
# Basic training with default config
uv run python experiments/train_model.py

# Train with specific experiment config
uv run python experiments/train_model.py experiment=train_peint

# Override specific parameters
uv run python experiments/train_model.py trainer.max_epochs=50 model.optimizer.lr=0.001

# Multiple parameter overrides
uv run python experiments/train_model.py experiment=train_peint \
    trainer.max_epochs=100 \
    data.batch_size=32 \
    model.weight_decay=0.1
```

### SLURM Cluster Training

For training on SLURM clusters:

```bash
# Submit job with default config
sbatch train.sh experiments/train_model.py experiment=train_peint

# Submit with parameter overrides
sbatch train.sh experiments/train_model.py experiment=train_peint trainer.max_epochs=200
```

### Standalone Training Scripts

For simpler experiments, use standalone scripts in `experiments/`:

```bash
# Fine-tune ESM2 model
uv run python experiments/finetune_esm.py

# Custom training logic
uv run python experiments/eval_model.py
```

## Configuration System

### Understanding Hydra Configs

The configuration system uses Hydra with a hierarchical structure:

- **Base config**: `configs/train.yaml` - sets default components
- **Component configs**: Individual modules (data, model, trainer, etc.)
- **Experiment configs**: `configs/experiment/` - complete experiment setups

### Key Configuration Files

- `configs/experiment/train_peint.yaml` - PEINT model training
- `configs/experiment/train_pipet.yaml` - PIPET model training
- `configs/data/plmr.yaml` - Data loading configuration
- `configs/model/peint.yaml` - PEINT model configuration

### Custom Experiments

Create new experiment configs in `configs/experiment/`:

```yaml
# @package _global_
defaults:
  - override /data: plmr
  - override /model: peint
  - override /trainer: gpu

tags: ["my_experiment"]
seed: 42

trainer:
  max_epochs: 100

model:
  optimizer:
    lr: 0.0003
```

## Available Models

### PEINT (Primary Model)
- Protein Evolution INference Transformer
- Configuration: `configs/model/peint.yaml`
- Network: `configs/net/peint.yaml`

### ESM2 Integration
- Pre-trained protein language model
- Fine-tuning support via `experiments/finetune_esm.py`

## Data Management

### Dataset Structure
- Store datasets in `./data/` directory
- Add processing instructions to `data/README.md`
- Use configuration files in `configs/data/` for data modules

### Supported Data Formats
- FASTA sequences
- Multiple sequence alignments (MSA)
- Custom dataset formats via data modules

## Development Workflow

### Code Structure
- Follow existing patterns in `peint/` package
- Use Lightning modules for models (`peint/models/`)
- Implement data modules in `peint/data/`
- Add utilities to `peint/utils/`

### Adding New Components

1. **New Model**: Add to `peint/models/` and create config in `configs/model/`
2. **New Dataset**: Implement in `peint/data/` and add config in `configs/data/`
3. **New Network**: Add to `peint/models/nets/` and create config in `configs/net/`

### Testing and Validation

**Test Organization Guidelines:**
- **NEVER** create test files in the project root directory
- All tests MUST be placed in the `tests/` folder using proper pytest structure
- Use descriptive test file names starting with `test_` (e.g., `test_pipet_pipeline.py`)
- Organize tests into classes and use pytest fixtures for setup
- Follow pytest naming conventions for test functions (`test_*`)

```bash
# Run pytest test suite
uv run pytest tests/

# Run specific test file
uv run pytest tests/test_pipet_pipeline.py

# Run tests with verbose output
uv run pytest tests/ -v

# Run validation scripts
uv run python scripts/validate_new_ckpt.py
uv run python scripts/validate_old_ckpt.py

# Inference and mutation analysis
uv run python scripts/infer_mutations.py
uv run python scripts/sample_on_tree.py
```

**Test File Structure:**
```
tests/
  test_pipet_pipeline.py      # PIPET training pipeline tests
  test_pipet_core.py          # Core PIPET functionality tests
  test_encoder_attention.py   # Multi-sequence attention tests
  test_encoder_simple.py      # Basic encoder functionality tests
  test_generative_metrics.py  # Existing generative metrics tests
```

## Notebooks and Analysis

Jupyter notebooks in `notebooks/` provide:
- `peint_inference.ipynb` - Model inference examples
- `zero_shot.ipynb` - Zero-shot evaluation
- `sample_on_tree.ipynb` - Phylogenetic sampling
- `validate_indels.ipynb` - Indel validation

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
- **Metrics**: Custom metrics in `peint/metrics/`

## Tips for New Developers

1. **Start Simple**: Use existing experiment configs before creating custom ones
2. **Check Examples**: Look at `notebooks/` for usage patterns
3. **Use Hydra**: Leverage configuration composition for reproducible experiments
4. **Monitor Training**: Set up logging early in your development process
5. **Test Locally**: Use small datasets and short training runs for development

## Troubleshooting

- **Import errors**: Ensure virtual environment is activated and dependencies installed
- **CUDA issues**: Check PyTorch installation and GPU availability
- **Configuration errors**: Use `--cfg job` to validate configs before training
- **Memory issues**: Reduce batch size in data configuration
- **Slow training**: Consider using multiple GPUs or gradient accumulation
