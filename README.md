# Evolutionary Models of Biological Sequence

This repository is a streamlined version of [protein-evolution](https://github.com/songlab-cal/protein-evolution) that aims to supports modality agnostic development of biological sequence evolution models in the PEINT framework.

## Setup

We use a submodule `evo` to store miscellaneous utility methods such as dataloading, tokenization, and msa processing to name a few. Any method that is not specific to evolution modeling, but is still useful for computational biology research should go into `evo`.

```bash
# After cloning the repository, please pull the evo submodule
git submodule update --init
```

We use [uv](https://docs.astral.sh/uv/getting-started/installation/) for python package management. Please follow the installation instructions for your machine before continuing.

```bash
# Create a venv for this project using python 3.10
uv venv --python 3.10

# We use cuda/12.8 for this project, but other versions should be ok
module load cuda/12.8

# Activate the venv
source .venv/bin/activate

# Sync the dependencies for all projects
uv sync

# Install development dependencies
uv pip install pre-commit
pre-commit install
```

## Data

All datasets should be stored inside the `./data` folder. Please push data processing scripts to github, but ignore large or sensitive files. Instead, please update the `data/README.md` file with instructions on how to procure the datasets that you are working with.

## Usage

We use [lightning](https://lightning.ai/docs/pytorch/stable/) to train models and [hydra](https://hydra.cc/docs/intro/) for dynamic config injection. The repository is inspired from this popular [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template) which offers a more detailed explanation of all the supported features. For the purposes of using this codebase, the following introduction should however be self-contained.

### Code Structure

```
peint/
├── configs/           # Hydra configuration files
│   ├── train.yaml     # Main training config
│   ├── experiment/    # Pre-built experiment setups
│   ├── data/          # Dataset configurations
│   ├── model/         # Model configurations
│   └── trainer/       # Training loop settings
├── peint/             # Core Python package
│   ├── models/        # Lightning modules and model implementations
│   ├── data/          # Data loaders and preprocessing
│   ├── utils/         # Helper functions and utilities
│   └── metrics/       # Custom evaluation metrics
├── experiments/       # Training scripts
│   ├── train_model.py # Main Hydra-based training
│   └── finetune_esm.py # Standalone ESM2 fine-tuning
├── scripts/           # Inference and validation tools
└── notebooks/         # Analysis and exploration
```

**Key Components:**
- **Hydra configs** allow you to mix and match components (data + model + trainer)
- **Lightning modules** handle training loops, optimization, and checkpointing
- **Experiments** provide both structured (Hydra) and simple training scripts

### Training Models

#### Quick Start - Train PEINT Model

```bash
# Train with the recommended PEINT configuration
uv run python experiments/train_model.py experiment=train_peint

# Train on multiple GPUs
uv run python experiments/train_model.py experiment=train_peint trainer=ddp trainer.devices=4
```

#### Customizing Training

The power of Hydra is that you can override any configuration parameter:

```bash
# Change learning rate and batch size
uv run python experiments/train_model.py experiment=train_peint \
    model.optimizer.lr=0.001 \
    data.batch_size=32

# Train for longer with different validation frequency
uv run python experiments/train_model.py experiment=train_peint \
    trainer.max_epochs=200 \
    trainer.val_check_interval=2000

# Resume from checkpoint
uv run python experiments/train_model.py experiment=train_peint \
    ckpt_path=/path/to/your/checkpoint.ckpt
```

#### SLURM Cluster Training

For training on computing clusters:

```bash
# Submit job to SLURM
sbatch train.sh experiments/train_model.py experiment=train_peint

# Submit with custom parameters
sbatch train.sh experiments/train_model.py experiment=train_peint trainer.max_epochs=500
```

#### Available Models

- **PEINT**: Protein evolution transformer (use `experiment=train_peint`)
- **PIPET**: Alternative model architecture (use `experiment=train_pipet`)
- **ESM2 Fine-tuning**: Standalone script at `experiments/finetune_esm.py`

### Adding Custom Experiments

Sometimes, the heavy templating associated with the `lightning-hydra` framework is not appropriate for the experiment that you'll want to run. In this case, it is perfectly acceptable to write a new training script inside the `./experiments` folder that suits your needs. For example, I wanted to finetune ESM2 using a lightweight training script and chose to create a standalone training scripts `experiments/finetune_esm.py` instead of using the hydra + lightning setup, which would have taken longer to setup. However, for important models that you expect to work with extensively and parameter tune, it is recommended to use the hydra framework which will save you time down the line.
