# CoSiNE Model Checkpoints

This directory contains pre-trained model checkpoints for the CoSiNE neural CTMC model.

## Available Checkpoints

### Primary Model: CoSiNE on DASM

The main model checkpoint trained on the DASM dataset for antibody sequence evolution.

**TODO: Add download link**

```bash
# Download pre-trained CoSiNE checkpoint
mkdir -p checkpoints
wget [TODO: ADD DOWNLOAD LINK] -O checkpoints/cosine_dasm.ckpt
```

**Model Details:**
- **Dataset**: DASM (Deep Antibody Sequence Modeling)
- **Training**: TODO: Add training details (epochs, batch size, etc.)
- **Architecture**: Neural CTMC with ESM-2 encoder
- **Rate Matrix**: Reversible (Pande transformation)
- **Performance**: TODO: Add validation metrics

### Checkpoint Files

```
checkpoints/
├── cosine_dasm.ckpt           # Main model checkpoint
├── cosine_dasm_best.ckpt      # Best validation checkpoint
└── cosine_dasm_last.ckpt      # Last training checkpoint
```

## Using Checkpoints

### Loading for Evaluation

Load a checkpoint for evaluation or inference:

```python
from cosine.models.modules.ctmc_module import CTMCModule
import torch

# Load checkpoint
checkpoint_path = "checkpoints/cosine_dasm.ckpt"
model = CTMCModule.load_from_checkpoint(checkpoint_path)
model.eval()

# Use the model
with torch.no_grad():
    outputs = model(x, y, t, x_sizes, y_sizes, chain_ids)
```

### Resuming Training

Resume training from a checkpoint:

```bash
# Resume training from checkpoint
uv run python experiments/train_model.py \
    experiment=train_ctmc_model_on_dasm \
    ckpt_path=checkpoints/cosine_dasm.ckpt
```

### Running Guided Sampling

Use a checkpoint for oracle-guided sampling:

```bash
# Run CoSiNE guided sampling with checkpoint
python scripts/guidance/cosine.py \
    --checkpoint checkpoints/cosine_dasm.ckpt \
    --oracle covid \
    --num_sequences 100 \
    --guidance_scale 2.0 \
    --output results/guided_sequences.csv
```

## Checkpoint Format

CoSiNE checkpoints are PyTorch Lightning checkpoint files containing:

- **Model state dict**: All model parameters
- **Optimizer state**: For resuming training
- **Hyperparameters**: Model configuration
- **Training metadata**: Epoch, global step, best validation score

Example checkpoint structure:

```python
checkpoint = {
    'state_dict': {...},           # Model parameters
    'optimizer_states': {...},     # Optimizer state
    'hyper_parameters': {
        'reversible': True,
        'embed_dim': 1280,
        'num_heads': 20,
        ...
    },
    'epoch': 50,
    'global_step': 100000,
}
```

## Checkpoint Information

### cosine_dasm.ckpt

**Model Configuration:**
```yaml
# Network architecture
encoder: ESM-2 (TODO: specify size)
embed_dim: TODO
num_heads: TODO
hidden_dim: TODO

# Training configuration
optimizer: AdamW
learning_rate: TODO
batch_size: TODO
max_epochs: TODO

# CTMC-specific
reversible: true
valid_tokens: 20 amino acids
```

**Training Details:**
- **Training time**: TODO
- **Hardware**: TODO (e.g., 4x A100 GPUs)
- **Dataset size**: TODO
- **Validation metrics**: TODO

**Performance:**
- **Validation likelihood**: TODO
- **Perplexity**: TODO
- **Guided sampling success rate**: TODO

## Converting Between Formats

### Extract Model Weights Only

If you only need model weights (without optimizer state):

```python
import torch

# Load checkpoint
checkpoint = torch.load("checkpoints/cosine_dasm.ckpt")

# Extract state dict
state_dict = checkpoint['state_dict']

# Save weights only
torch.save(state_dict, "checkpoints/cosine_weights.pt")
```

### Loading in Custom Code

Load checkpoint weights into a custom model:

```python
from cosine.models.nets.ctmc import NeuralCTMC
import torch

# Create model
model = NeuralCTMC(...)

# Load checkpoint state dict
checkpoint = torch.load("checkpoints/cosine_dasm.ckpt")
model.load_state_dict(checkpoint['state_dict'], strict=False)
```

## Checkpoint Management

### During Training

Checkpoints are automatically saved during training based on:

1. **Best validation score**: Saved when validation improves
2. **Every N epochs**: Periodic checkpoints
3. **Last checkpoint**: Most recent training state

Configure in `configs/callbacks/model_checkpoint.yaml`:

```yaml
_target_: lightning.pytorch.callbacks.ModelCheckpoint
dirpath: ${paths.output_dir}/checkpoints
filename: "epoch_{epoch:03d}_val_{val/loss:.4f}"
monitor: "val/loss"
mode: "min"
save_top_k: 3
save_last: true
```

### Storage Recommendations

- **Git**: Do NOT commit checkpoint files to git (too large)
- **Cloud**: Upload to cloud storage (e.g., Google Drive, AWS S3)
- **Sharing**: Use download links in this README
- **Backup**: Keep multiple versions during development

## Checkpoint Verification

Verify checkpoint integrity before use:

```python
import torch

def verify_checkpoint(checkpoint_path):
    """Verify that a checkpoint can be loaded."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print(f"✓ Checkpoint loaded successfully")
        print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  Global step: {checkpoint.get('global_step', 'N/A')}")
        print(f"  Keys: {list(checkpoint.keys())}")
        return True
    except Exception as e:
        print(f"✗ Failed to load checkpoint: {e}")
        return False

verify_checkpoint("checkpoints/cosine_dasm.ckpt")
```

## Citation

If you use CoSiNE pre-trained checkpoints, please cite:

```bibtex
@article{cosine2024,
  title={CoSiNE: Conditional Simulation of Evolutionary Sequences with Neural CTMC Models},
  author={Your Name and Collaborators},
  journal={TODO: Add journal/preprint},
  year={2024}
}
```

## Contact

For questions about checkpoints:
- Open an issue on GitHub
- Contact: stephen.lu@berkeley.edu

## Notes

- Checkpoints are compatible with PyTorch Lightning 2.0+
- Models were trained with Python 3.10 and PyTorch 2.6+
- GPU recommended for inference (CPU supported but slower)
- Checkpoint files are typically 500MB - 2GB in size
