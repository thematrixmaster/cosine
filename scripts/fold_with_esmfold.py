#!/usr/bin/env python
"""
Fold sequences using ESMFold in enroot container and extract pLDDT scores.
"""

import argparse
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


def analyze_branch_lengths(csv_path: Path) -> pd.DataFrame:
    """Read CSV and return branch_length statistics."""
    df = pd.read_csv(csv_path)
    print(f"\n{csv_path.name}:")
    print(f"  Total rows: {len(df)}")
    print(
        f"  Branch length range: {df['branch_length'].min():.4f} - {df['branch_length'].max():.4f}"
    )
    print(f"  Branch length mean: {df['branch_length'].mean():.4f}")
    print(f"  Branch length median: {df['branch_length'].median():.4f}")
    return df


def stratified_subsample(df: pd.DataFrame, n_samples: int, n_bins: int = 10) -> np.ndarray:
    """Create stratified subsample based on branch_length distribution."""
    df["branch_bin"] = pd.qcut(df["branch_length"], q=n_bins, labels=False, duplicates="drop")

    samples_per_bin = n_samples // n_bins
    remainder = n_samples % n_bins

    sampled_indices = []
    for bin_idx in range(n_bins):
        bin_df = df[df["branch_bin"] == bin_idx]
        n_from_bin = samples_per_bin + (1 if bin_idx < remainder else 0)
        n_from_bin = min(n_from_bin, len(bin_df))

        sampled = bin_df.sample(n=n_from_bin, random_state=42)
        sampled_indices.extend(sampled.index.tolist())

    return np.array(sorted(sampled_indices))


def create_fasta(sequences: list, seq_ids: list, fasta_path: Path):
    """Create a FASTA file from sequences."""
    with open(fasta_path, "w") as f:
        for seq_id, seq in zip(seq_ids, sequences):
            f.write(f">{seq_id}\n{seq}\n")


def run_esmfold(fasta_path: Path, output_dir: Path):
    """
    Run ESMFold using enroot container.
    Paths must be relative to the peint directory since it's mounted at /home.
    """
    # Convert absolute paths to paths relative to peint directory
    peint_dir = Path("/accounts/projects/yss/stephen.lu/peint")

    # Get relative paths
    rel_fasta = fasta_path.relative_to(peint_dir)
    rel_output = output_dir.relative_to(peint_dir)

    # Paths inside container (mounted at /home)
    container_fasta = f"/home/{rel_fasta}"
    container_output = f"/home/{rel_output}"

    cmd = [
        "enroot",
        "start",
        "--root",
        "-w",
        "--mount",
        f'{os.environ["HOME"]}/peint:/home',
        "esmfold",
        "-i",
        container_fasta,
        "-o",
        container_output,
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
    """Extract mean pLDDT from PDB file (B-factor column)."""
    plddt_values = []

    with open(pdb_path, "r") as f:
        for line in f:
            if line.startswith("ATOM"):
                try:
                    # B-factor is in columns 61-66
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
        # ESMFold outputs PDB files with the sequence ID as filename
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
    parser = argparse.ArgumentParser(description="Subsample and fold sequences with ESMFold")
    parser.add_argument("--n_samples", type=int, default=100, help="Number of samples per CSV")
    parser.add_argument(
        "--n_bins", type=int, default=10, help="Number of bins for stratified sampling"
    )
    parser.add_argument("--analyze_only", action="store_true", help="Only analyze, do not fold")
    args = parser.parse_args()

    # Paths
    results_dir = Path("/accounts/projects/yss/stephen.lu/peint/results/gen_eval")
    work_dir = Path("/accounts/projects/yss/stephen.lu/peint/results/esmfold_folding")
    work_dir.mkdir(exist_ok=True, parents=True)

    csv_files = [
        results_dir / "peint.csv",
        results_dir / "ctmc_gillespie.csv",
        results_dir / "ctmc_mat_exp.csv",
    ]

    # Analyze distributions
    print("=" * 80)
    print("ANALYZING BRANCH LENGTH DISTRIBUTIONS")
    print("=" * 80)

    dfs = {}
    for csv_file in csv_files:
        df = analyze_branch_lengths(csv_file)
        dfs[csv_file.stem] = df

    # Create stratified subsample
    print(f"\n{'=' * 80}")
    print(f"CREATING STRATIFIED SUBSAMPLE (n={args.n_samples}, bins={args.n_bins})")
    print("=" * 80)

    ref_df = dfs["peint"]
    sampled_indices = stratified_subsample(ref_df, args.n_samples, args.n_bins)

    print(f"\nSelected {len(sampled_indices)} rows")
    print(
        f"Sampled branch_length range: {ref_df.iloc[sampled_indices]['branch_length'].min():.4f} - {ref_df.iloc[sampled_indices]['branch_length'].max():.4f}"
    )

    if args.analyze_only:
        print("\nAnalysis complete (--analyze_only flag set). Exiting.")
        return

    # Fold sequences for each CSV
    print(f"\n{'=' * 80}")
    print("FOLDING SEQUENCES WITH ESMFOLD")
    print("=" * 80)

    for csv_name, df in dfs.items():
        print(f"\n### Processing {csv_name} ###")

        subsampled_df = df.iloc[sampled_indices]

        # Fold heavy chains
        print("\n>> Folding heavy chains...")
        hv_dir = work_dir / f"{csv_name}_hv"
        hv_dir.mkdir(exist_ok=True)

        hv_fasta = hv_dir / "sequences.fasta"
        hv_sequences = subsampled_df["sim_child_hv"].tolist()
        hv_ids = [f"{csv_name}_hv_{i}" for i in range(len(hv_sequences))]

        create_fasta(hv_sequences, hv_ids, hv_fasta)
        run_esmfold(hv_fasta, hv_dir)
        hv_plddts = extract_plddts(hv_dir, hv_ids)

        # Fold light chains
        print("\n>> Folding light chains...")
        lt_dir = work_dir / f"{csv_name}_lt"
        lt_dir.mkdir(exist_ok=True)

        lt_fasta = lt_dir / "sequences.fasta"
        lt_sequences = subsampled_df["sim_child_lt"].tolist()
        lt_ids = [f"{csv_name}_lt_{i}" for i in range(len(lt_sequences))]

        create_fasta(lt_sequences, lt_ids, lt_fasta)
        run_esmfold(lt_fasta, lt_dir)
        lt_plddts = extract_plddts(lt_dir, lt_ids)

        # Add pLDDT scores to dataframe
        df["sim_child_hv_plddt"] = np.nan
        df["sim_child_lt_plddt"] = np.nan

        for i, idx in enumerate(sampled_indices):
            hv_id = hv_ids[i]
            lt_id = lt_ids[i]
            df.loc[idx, "sim_child_hv_plddt"] = hv_plddts.get(hv_id, 0.0)
            df.loc[idx, "sim_child_lt_plddt"] = lt_plddts.get(lt_id, 0.0)

        # Save updated CSV
        csv_file = results_dir / f"{csv_name}.csv"
        df.to_csv(csv_file, index=False)

        # Report statistics
        valid_hv = df.loc[sampled_indices, "sim_child_hv_plddt"].dropna()
        valid_lt = df.loc[sampled_indices, "sim_child_lt_plddt"].dropna()

        print(f"\n✓ Updated {csv_file.name}")
        print(f"  Rows with pLDDT scores: {len(sampled_indices)}")
        print(f"  Mean pLDDT (HV): {valid_hv.mean():.2f} ± {valid_hv.std():.2f}")
        print(f"  Mean pLDDT (LT): {valid_lt.mean():.2f} ± {valid_lt.std():.2f}")

    print(f"\n{'=' * 80}")
    print("COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
