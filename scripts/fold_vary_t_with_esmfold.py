#!/usr/bin/env python
"""
Fold sequences from vary_t CSV files using ESMFold and extract pLDDT scores.
These files have columns: hv_seqs, lt_seqs
"""

import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


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
    # Paths
    results_dir = Path("/accounts/projects/yss/stephen.lu/peint/results/gen_eval")
    work_dir = Path("/accounts/projects/yss/stephen.lu/peint/results/esmfold_vary_t")
    work_dir.mkdir(exist_ok=True, parents=True)

    csv_files = [
        results_dir / "peint_vary_t.csv",
        results_dir / "ctmc_gillespie_vary_t.csv",
        results_dir / "ctmc_mat_exp_vary_t.csv",
    ]

    print("=" * 80)
    print("FOLDING VARY_T SEQUENCES WITH ESMFOLD")
    print("=" * 80)

    for csv_file in csv_files:
        csv_name = csv_file.stem
        print(f"\n### Processing {csv_name} ###")

        # Read CSV
        df = pd.read_csv(csv_file)
        print(f"  Total rows: {len(df)}")

        # Fold heavy chains (hv_seqs)
        print("\n>> Folding heavy chains...")
        hv_dir = work_dir / f"{csv_name}_hv"
        hv_dir.mkdir(exist_ok=True)

        hv_fasta = hv_dir / "sequences.fasta"
        hv_sequences = df["hv_seqs"].tolist()
        hv_ids = [f"{csv_name}_hv_{i}" for i in range(len(hv_sequences))]

        create_fasta(hv_sequences, hv_ids, hv_fasta)
        run_esmfold(hv_fasta, hv_dir)
        hv_plddts = extract_plddts(hv_dir, hv_ids)

        # Fold light chains (lt_seqs)
        print("\n>> Folding light chains...")
        lt_dir = work_dir / f"{csv_name}_lt"
        lt_dir.mkdir(exist_ok=True)

        lt_fasta = lt_dir / "sequences.fasta"
        lt_sequences = df["lt_seqs"].tolist()
        lt_ids = [f"{csv_name}_lt_{i}" for i in range(len(lt_sequences))]

        create_fasta(lt_sequences, lt_ids, lt_fasta)
        run_esmfold(lt_fasta, lt_dir)
        lt_plddts = extract_plddts(lt_dir, lt_ids)

        # Add pLDDT scores to dataframe
        df["hv_seqs_plddt"] = [hv_plddts.get(hv_ids[i], 0.0) for i in range(len(df))]
        df["lt_seqs_plddt"] = [lt_plddts.get(lt_ids[i], 0.0) for i in range(len(df))]

        # Save updated CSV
        df.to_csv(csv_file, index=False)

        # Report statistics
        print(f"\n✓ Updated {csv_file.name}")
        print(f"  Rows with pLDDT scores: {len(df)}")
        print(
            f"  Mean pLDDT (HV): {df['hv_seqs_plddt'].mean():.2f} ± {df['hv_seqs_plddt'].std():.2f}"
        )
        print(
            f"  Mean pLDDT (LT): {df['lt_seqs_plddt'].mean():.2f} ± {df['lt_seqs_plddt'].std():.2f}"
        )

    print(f"\n{'=' * 80}")
    print("COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
