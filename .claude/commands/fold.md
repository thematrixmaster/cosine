# Protein Folding Pipeline with ESMFold

This guide documents the pipeline for folding protein sequences using ESMFold and extracting pLDDT scores.

## Overview

We use **ESMFold via enroot container** to fold sequences and extract predicted Local Distance Difference Test (pLDDT) scores, which indicate structural confidence. The pLDDT scores are extracted from the B-factor column (positions 61-66) of the output PDB files.

## Available Scripts

### 1. `scripts/fold_with_esmfold.py`
**Use case:** Large datasets where you want stratified subsampling based on `branch_length` distribution

**Features:**
- Stratified sampling using quantile-based binning on `branch_length` column
- Processes sequences from columns: `sim_child_hv` and `sim_child_lt`
- Adds pLDDT scores to columns: `sim_child_hv_plddt` and `sim_child_lt_plddt`
- Only updates sampled rows (rest get NaN values)

**Command:**
```bash
uv run python scripts/fold_with_esmfold.py --n_samples 100 --n_bins 10
```

**Parameters:**
- `--n_samples`: Number of sequences to sample per CSV file (default: 100)
- `--n_bins`: Number of bins for stratified sampling (default: 10)
- `--analyze_only`: Only analyze distributions, don't fold

**Expected input CSV files:**
- `results/gen_eval/peint.csv`
- `results/gen_eval/ctmc_gillespie.csv`
- `results/gen_eval/ctmc_mat_exp.csv`

**Required columns:**
- `sim_child_hv`: Heavy chain sequences
- `sim_child_lt`: Light chain sequences
- `branch_length`: Used for stratified sampling

**Output:**
- Adds `sim_child_hv_plddt` and `sim_child_lt_plddt` columns
- Creates PDB structures in `results/esmfold_folding/`

### 2. `scripts/fold_vary_t_with_esmfold.py`
**Use case:** Smaller datasets where you want to fold ALL sequences (no sampling)

**Features:**
- Processes all rows in the CSV files
- Processes sequences from columns: `hv_seqs` and `lt_seqs`
- Adds pLDDT scores to columns: `hv_seqs_plddt` and `lt_seqs_plddt`
- All rows get pLDDT scores

**Command:**
```bash
uv run python scripts/fold_vary_t_with_esmfold.py
```

**Expected input CSV files:**
- `results/gen_eval/peint_vary_t.csv`
- `results/gen_eval/ctmc_gillespie_vary_t.csv`
- `results/gen_eval/ctmc_mat_exp_vary_t.csv`

**Required columns:**
- `hv_seqs`: Heavy chain sequences
- `lt_seqs`: Light chain sequences

**Output:**
- Adds `hv_seqs_plddt` and `lt_seqs_plddt` columns
- Creates PDB structures in `results/esmfold_vary_t/`

## Core ESMFold Command

The underlying ESMFold command via enroot container:

```bash
enroot start --root -w \
  --mount ${HOME}/peint:/home \
  esmfold \
  -i /home/path/to/sequences.fasta \
  -o /home/path/to/output_dir/
```

**Critical details:**
- Mount the peint directory at `/home` in the container
- ALL paths inside the container must be relative to `/home`
- Convert absolute paths to container paths:
  ```python
  peint_dir = Path('/accounts/projects/yss/stephen.lu/peint')
  rel_path = absolute_path.relative_to(peint_dir)
  container_path = f"/home/{rel_path}"
  ```

## Creating Custom Folding Scripts

If you need to fold sequences from a different CSV format, create a new script following this pattern:

### Template Structure

```python
#!/usr/bin/env python
"""
Fold sequences using ESMFold and extract pLDDT scores.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import subprocess
import os

def create_fasta(sequences: list, seq_ids: list, fasta_path: Path):
    """Create a FASTA file from sequences."""
    with open(fasta_path, 'w') as f:
        for seq_id, seq in zip(seq_ids, sequences):
            f.write(f">{seq_id}\n{seq}\n")

def run_esmfold(fasta_path: Path, output_dir: Path):
    """
    Run ESMFold using enroot container.
    CRITICAL: Paths must be relative to peint directory since it's mounted at /home.
    """
    peint_dir = Path('/accounts/projects/yss/stephen.lu/peint')

    # Convert to container paths
    rel_fasta = fasta_path.relative_to(peint_dir)
    rel_output = output_dir.relative_to(peint_dir)

    container_fasta = f"/home/{rel_fasta}"
    container_output = f"/home/{rel_output}"

    cmd = [
        'enroot', 'start',
        '--root', '-w',
        '--mount', f'{os.environ["HOME"]}/peint:/home',
        'esmfold',
        '-i', container_fasta,
        '-o', container_output
    ]

    print(f"Running ESMFold...")
    print(f"Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"ESMFold failed with return code {result.returncode}")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        raise RuntimeError("ESMFold failed")

    print(result.stdout)

def extract_plddt_from_pdb(pdb_path: Path) -> float:
    """
    Extract mean pLDDT from PDB file.
    pLDDT is stored in the B-factor column (positions 61-66) of ATOM lines.
    """
    plddt_values = []

    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith('ATOM'):
                try:
                    # B-factor is in columns 61-66 (PDB format specification)
                    plddt = float(line[60:66].strip())
                    plddt_values.append(plddt)
                except (ValueError, IndexError):
                    continue

    if plddt_values:
        return np.mean(plddt_values)
    else:
        return 0.0

def extract_plddts(output_dir: Path, seq_ids: list) -> dict:
    """Extract pLDDT scores from ESMFold output PDB files."""
    plddt_scores = {}

    for seq_id in seq_ids:
        pdb_file = output_dir / f"{seq_id}.pdb"

        if pdb_file.exists():
            plddt = extract_plddt_from_pdb(pdb_file)
            plddt_scores[seq_id] = plddt
            print(f"  {seq_id}: pLDDT = {plddt:.2f}")
        else:
            print(f"  Warning: PDB file not found for {seq_id}")
            plddt_scores[seq_id] = 0.0

    return plddt_scores

def main():
    # TODO: Customize this section for your CSV format

    # Define paths
    csv_file = Path('/path/to/your/sequences.csv')
    work_dir = Path('/accounts/projects/yss/stephen.lu/peint/results/esmfold_custom')
    work_dir.mkdir(exist_ok=True, parents=True)

    # Read CSV
    df = pd.read_csv(csv_file)

    # Extract sequences (CUSTOMIZE COLUMN NAMES)
    sequences = df['your_sequence_column'].tolist()
    seq_ids = [f"seq_{i}" for i in range(len(sequences))]

    # Create output directory
    output_dir = work_dir / "structures"
    output_dir.mkdir(exist_ok=True)

    # Create FASTA file
    fasta_path = output_dir / "sequences.fasta"
    create_fasta(sequences, seq_ids, fasta_path)

    # Run ESMFold
    run_esmfold(fasta_path, output_dir)

    # Extract pLDDT scores
    plddt_scores = extract_plddts(output_dir, seq_ids)

    # Add to dataframe (CUSTOMIZE COLUMN NAME)
    df['plddt'] = [plddt_scores.get(seq_ids[i], 0.0) for i in range(len(df))]

    # Save updated CSV
    df.to_csv(csv_file, index=False)

    print(f"\n Updated {csv_file.name}")
    print(f"  Mean pLDDT: {df['plddt'].mean():.2f} ± {df['plddt'].std():.2f}")

if __name__ == '__main__':
    main()
```

## Interpreting pLDDT Scores

pLDDT (predicted Local Distance Difference Test) scores range from 0-100:

- **90-100**: Very high confidence - well-structured regions
- **70-90**: High confidence - generally good structure
- **50-70**: Low confidence - potentially flexible or disordered
- **<50**: Very low confidence - likely disordered or incorrect

### Expected ranges for antibody sequences:
- **Well-folded sequences**: 80-90 (typical for PEINT and CTMC matrix exponential)
- **Light chains**: Typically score 3-5 points higher than heavy chains
- **Problematic sequences**: <70 with high variation (like CTMC Gillespie: 73.47 ± 11.98)

## Common Use Cases

### Case 1: Fold large dataset with sampling
```bash
# Uses stratified sampling on branch_length
uv run python scripts/fold_with_esmfold.py --n_samples 100 --n_bins 10

# Analyze only (no folding)
uv run python scripts/fold_with_esmfold.py --n_samples 100 --analyze_only
```

### Case 2: Fold all sequences in smaller dataset
```bash
# Processes all rows
uv run python scripts/fold_vary_t_with_esmfold.py
```

### Case 3: Create custom script for new CSV format
1. Copy `scripts/fold_vary_t_with_esmfold.py` to new file
2. Modify the following sections:
   - CSV file paths (lines 96-103)
   - Column names for sequences (lines 124, 137)
   - Output column names (lines 145-146)
   - Output directory path (line 97)
3. Run the script

## Running in Background

To run folding tasks in the background and monitor progress:

```bash
# Start in background, log to file
uv run python scripts/fold_with_esmfold.py --n_samples 100 2>&1 | tee /tmp/esmfold_run.log &

# Monitor progress
tail -f /tmp/esmfold_run.log

# Check specific sections
grep "Mean pLDDT" /tmp/esmfold_run.log
```

## Performance Notes

- **Speed**: ~0.6-0.7s per sequence (batched)
- **Batch sizes**: 8-9 sequences per batch (automatic)
- **Typical runtime**:
  - 100 sequences: ~2 minutes
  - 300 sequences: ~6 minutes
  - 600 sequences: ~8 minutes

## Troubleshooting

### Problem: ESMFold can't find input files
**Solution**: Ensure paths are converted to container paths (relative to `/home`)

### Problem: Permission denied
**Solution**: Use `--root` flag in enroot command (already in scripts)

### Problem: Out of memory
**Solution**: Process in smaller batches or reduce number of sequences

### Problem: Low pLDDT scores
**Causes**:
- Invalid sequences (check for non-standard amino acids)
- Very long sequences (>500 residues)
- Model generated unnatural sequences

## Example Results

### Successful run (peint_vary_t.csv):
```
 Updated peint_vary_t.csv
  Rows with pLDDT scores: 100
  Mean pLDDT (HV): 85.35 ± 1.29
  Mean pLDDT (LT): 83.46 ± 7.56
```

### Problematic run (ctmc_gillespie_vary_t.csv):
```
 Updated ctmc_gillespie_vary_t.csv
  Rows with pLDDT scores: 100
  Mean pLDDT (HV): 73.47 ± 11.98  ê Low mean, high variation
  Mean pLDDT (LT): 80.88 ± 8.74
```

## Files Created

After running the scripts, you'll find:

```
results/esmfold_folding/          # From fold_with_esmfold.py
   peint_hv/
      sequences.fasta
      *.pdb                      # One PDB per sequence
   peint_lt/
      ...
   ...

results/esmfold_vary_t/            # From fold_vary_t_with_esmfold.py
   peint_vary_t_hv/
      ...
   ...
```

## Quick Reference

| Task | Command | Time |
|------|---------|------|
| Fold 100 samples (stratified) | `uv run python scripts/fold_with_esmfold.py --n_samples 100` | ~2 min |
| Fold all vary_t sequences | `uv run python scripts/fold_vary_t_with_esmfold.py` | ~8 min |
| Analyze distributions only | `uv run python scripts/fold_with_esmfold.py --analyze_only` | <1 sec |

---

**Last updated:** 2025-11-05
**Scripts location:** `/accounts/projects/yss/stephen.lu/peint/scripts/`
