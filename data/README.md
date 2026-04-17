# CoSiNE Dataset Documentation

This directory contains datasets used for training and evaluating the CoSiNE model.

## DASM Dataset

The **DASM** (Deep Antibody Sequence Modeling) dataset is the primary dataset used for training the CoSiNE neural CTMC model.

### Overview

- **Purpose**: Training CoSiNE on evolutionary pairs of antibody sequences
- **Size**: TODO: Add dataset size information
- **Format**: Evolutionary sequence pairs with phylogenetic distances
- **Domain**: Antibody heavy and light chain sequences

### Download

**TODO: Add download link**

```bash
# Download DASM dataset
mkdir -p data/dasm
wget [TODO: ADD DOWNLOAD LINK] -O data/dasm.tar.gz
tar -xzf data/dasm.tar.gz -C data/
```

### Dataset Structure

The DASM dataset contains evolutionary pairs of antibody sequences:

```
data/dasm/
├── train/
│   ├── sequences.fasta      # Training sequence pairs
│   └── metadata.csv         # Phylogenetic distances, chain info
├── val/
│   ├── sequences.fasta      # Validation sequence pairs
│   └── metadata.csv
└── test/
    ├── sequences.fasta      # Test sequence pairs
    └── metadata.csv
```

### Data Format

Each entry in the dataset consists of:

- **Ancestor sequence (x)**: Parent antibody sequence
- **Descendant sequence (y)**: Child antibody sequence
- **Phylogenetic distance (t)**: Evolutionary time between sequences
- **Chain IDs**: Heavy (H) and light (L) chain identifiers
- **Sizes**: Sequence lengths for each chain

Example metadata row:

```csv
pair_id,ancestor_seq,descendant_seq,distance,heavy_len,light_len
PAIR_001,EVQL...QTV,EVQL...QTV,0.023,120,105
```

### Usage

The DASM dataset is configured in Hydra configs:

- **Training**: `configs/data/dataset/dasm-tr.yaml`
- **Validation**: `configs/data/dataset/dasm-val.yaml`

Load the dataset using:

```python
from cosine.data.datasets.dasm import DASMDataset

# Load training data
train_dataset = DASMDataset(
    data_dir="data/dasm/train",
    vocab=vocab,
    max_len=1022
)
```

### Data Processing

The DASM dataset preprocessing includes:

1. **Sequence pairing**: Matching ancestral and descendant sequences from phylogenetic trees
2. **Distance computation**: Calculating evolutionary distances from tree branch lengths
3. **Tokenization**: Converting amino acid sequences to token indices
4. **Padding**: Padding sequences to fixed length

See `data/dasm.py` for data processing scripts.

## Data Loading

The CoSiNE training pipeline uses the following data loading configuration:

```yaml
# configs/data/plmr.yaml
_target_: cosine.data.datamodule.PLMRDataModule
train_dataset:
  _target_: cosine.data.datasets.dasm.DASMDataset
  data_dir: ${paths.data_dir}/dasm/train
val_dataset:
  _target_: cosine.data.datasets.dasm.DASMDataset
  data_dir: ${paths.data_dir}/dasm/val
batch_size: 32
num_workers: 4
```

## Additional Datasets

While CoSiNE is primarily trained on DASM, the repository can be extended to other datasets:

### Adding New Datasets

1. Create dataset class in `cosine/data/datasets/your_dataset.py`
2. Implement `__getitem__` and `__len__` methods
3. Add configuration in `configs/data/dataset/your_dataset.yaml`
4. Document dataset structure and download instructions here

Example dataset class:

```python
from torch.utils.data import Dataset

class YourDataset(Dataset):
    def __init__(self, data_dir, vocab, max_len=1022):
        self.data_dir = data_dir
        self.vocab = vocab
        self.max_len = max_len
        # Load your data here

    def __getitem__(self, idx):
        # Return (x, y, t, x_sizes, y_sizes, chain_ids)
        pass

    def __len__(self):
        return len(self.data)
```

## Data Processing Scripts

Data processing utilities are available in:

- `data/dasm.py` - DASM dataset processing
- `data/oas.py` - OAS dataset processing
- `data/wyatt.py` - Wyatt dataset processing
- `data/cml.py` - CherryML data processing

## Important Notes

1. **Data Location**: Large datasets are stored in `data/` but not committed to git
2. **Symlinks**: The data directory may contain symlinks to external storage locations
3. **Preprocessing**: Raw data should be preprocessed before training
4. **Validation**: Always validate dataset integrity before training

## Citation

If you use the DASM dataset, please cite:

```bibtex
@article{dasm2024,
  title={TODO: Add DASM citation},
  author={TODO},
  journal={TODO},
  year={2024}
}
```

## Contact

For questions about datasets or data processing:
- Open an issue on GitHub
- Contact: stephen.lu@berkeley.edu
