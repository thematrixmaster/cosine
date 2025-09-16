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

# Activate the venv
source .venv/bin/activate

# Sync the dependencies for all projects
uv sync
```

## Usage

