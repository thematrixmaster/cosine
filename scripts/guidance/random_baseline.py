"""
Random Mutation Baseline

The simplest possible baseline: randomly mutate N positions within the masked region.
No optimization, no intelligence - just uniform random sampling.

This serves as a sanity check baseline to ensure that sophisticated methods
(HotFlip, Genetic, MLM, CTMC) are actually doing better than random chance.

Usage:
    uv run python scripts/random_baseline.py \
        --oracle SARSCoV2Beta \
        --oracle-seed-idx 0 \
        --max-mutations 5 \
        --mask-region CDR_overall \
        --batch-size 1000
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'evo'))

import argparse
import time
import random
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from datetime import datetime

from evo.oracles import get_oracle
from evo.antibody import create_region_masks, compute_oasis_humanness, compute_iglm_humanness
from oracle_counter import OracleCallCounter

# Standard amino acid vocabulary
AA_VOCAB = list("ACDEFGHIKLMNPQRSTVWY")

def parse_args():
    parser = argparse.ArgumentParser(description='Random Mutation Baseline')

    # Oracle & Seed
    parser.add_argument('--oracle', type=str, default='SARSCoV2Beta', choices=['SARSCoV1', 'SARSCoV2Beta'])
    parser.add_argument('--oracle-seed-idx', type=int, default=0)
    parser.add_argument('--seed-seq', type=str, default=None,
                        help='Custom seed sequence to optimize (overrides oracle-seed-idx if provided)')

    # Sampling Params
    parser.add_argument('--batch-size', type=int, default=100, help='Number of random sequences to generate')
    parser.add_argument('--max-mutations', type=int, default=5, help='Number of random mutations to apply')

    # Constraints
    parser.add_argument('--mask-region', type=str, default=None,
                        choices=['CDR1', 'CDR2', 'CDR3', 'CDR_overall', 'FR1', 'FR2', 'FR3', 'FR4', None])

    # Tech
    parser.add_argument('--save-csv', action='store_true', help='Whether to save output as CSV')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--output-path', type=str, default=None)

    return parser.parse_args()

# --- Helper Functions ---

def calculate_hamming_distance(seq1: str, seq2: str) -> int:
    """Calculate Hamming distance between two sequences of equal length."""
    if len(seq1) != len(seq2):
        raise ValueError("Sequences must be of equal length to compute Hamming distance.")
    return sum(c1 != c2 for c1, c2 in zip(seq1, seq2))

def calculate_internal_diversity(sequences: list[str]):
    """Calculate average pairwise Hamming distance within a list of sequences."""
    n = len(sequences)
    if n < 2:
        return 0.0, 0.0
    distances = []
    count = 0
    for i in tqdm(range(n)):
        for j in range(i + 1, n):
            distances.append(calculate_hamming_distance(sequences[i], sequences[j]))
            count += 1
    mean_distance = sum(distances) / count
    std_distance = np.std(distances)
    return mean_distance, std_distance

def load_oracle_binders(oracle_variant: str) -> pd.DataFrame:
    csv_files = {"SARSCoV1": "CovAbDab_heavy_binds SARS-CoV1.csv", "SARSCoV2Beta": "CovAbDab_heavy_binds SARS-CoV2_Beta.csv"}
    csv_path = Path("/accounts/projects/yss/stephen.lu/peint-workspace/main/evo/evo/oracles/data") / csv_files[oracle_variant]
    if not csv_path.exists():
        csv_path = Path(__file__).parent.parent / "evo/evo/oracles/data" / csv_files[oracle_variant]
    df = pd.read_csv(csv_path)
    df.columns = ['sequence', 'binding']
    return df[df['binding'] == 1].reset_index(drop=True)

def select_oracle_seed(oracle_variant: str, seed_idx: int) -> str:
    binders = load_oracle_binders(oracle_variant)
    return binders.iloc[seed_idx]['sequence']

def get_region_mask(sequence: str, region_name: str) -> np.ndarray:
    """Returns boolean mask where True = Mutable, False = Frozen."""
    if region_name is None:
        return np.ones(len(sequence), dtype=bool)
    masks = create_region_masks(sequence, scheme="imgt")
    return masks[region_name]

# --- Core Random Mutation Logic ---

def apply_random_mutations(
    seed_seq: str,
    n_mutations: int,
    mask: np.ndarray
) -> str:
    """
    Randomly mutate exactly n_mutations positions within the masked region.

    Parameters
    ----------
    seed_seq : str
        Starting sequence
    n_mutations : int
        Number of positions to mutate
    mask : np.ndarray
        Boolean mask (True = mutable positions)

    Returns
    -------
    str
        Mutated sequence
    """
    # Get mutable indices
    mutable_indices = np.where(mask)[0]

    if len(mutable_indices) == 0:
        return seed_seq

    # Cap n_mutations at the number of mutable positions
    n_mutations = min(n_mutations, len(mutable_indices))

    # Randomly select positions to mutate
    positions_to_mutate = np.random.choice(mutable_indices, size=n_mutations, replace=False)

    # Apply mutations
    seq_list = list(seed_seq)
    for pos in positions_to_mutate:
        current_aa = seq_list[pos]
        # Choose a different amino acid (avoid self-mutation)
        possible_aas = [aa for aa in AA_VOCAB if aa != current_aa]
        new_aa = random.choice(possible_aas)
        seq_list[pos] = new_aa

    return "".join(seq_list)

# --- Main ---

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Reproducibility
    seed = args.seed if args.seed is not None else int(time.time())
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    print(f"{'='*60}\nRunning Random Mutation Baseline\n{'='*60}")
    print(f"Mutations per sequence: {args.max_mutations}")
    print(f"Batch size: {args.batch_size}")

    # Load Oracle
    print(f"Loading Oracle: {args.oracle}")
    oracle = OracleCallCounter(get_oracle(
        args.oracle,
        enable_mc_dropout=True,
        mc_samples=10,
        mc_dropout_seed=42,
        use_iglm_weighting=False,
        device=device,
        precompute_seed_fitnesses=False
    ))

    # Load Seed
    if args.seed_seq is not None:
        seed_seq = args.seed_seq
        print(f"Using custom seed sequence ({len(seed_seq)} aa)")
    else:
        seed_seq = select_oracle_seed(args.oracle, args.oracle_seed_idx)
    seed_fitness = oracle.predict(seed_seq, increment=False)[0]
    print(f"Seed Fitness: {seed_fitness:.4f}")

    # Create Mask
    mask = get_region_mask(seed_seq, args.mask_region)
    if args.mask_region:
        print(f"Constraining mutations to {args.mask_region} ({mask.sum()} residues)")
    else:
        print(f"Mutations allowed at all {mask.sum()} positions")

    results = []

    print(f"\nGenerating {args.batch_size} random mutants...")
    start_time = time.time()

    for i in tqdm(range(args.batch_size)):
        traj_start_time = time.time()

        # Generate random mutant
        mutant_seq = apply_random_mutations(
            seed_seq=seed_seq,
            n_mutations=args.max_mutations,
            mask=mask
        )

        # Evaluate fitness
        mutant_fitness = oracle.predict(mutant_seq, increment=False)[0]

        traj_end_time = time.time()
        time_taken = traj_end_time - traj_start_time

        # Calculate actual number of mutations (should be exactly max_mutations)
        n_muts = calculate_hamming_distance(seed_seq, mutant_seq)

        results.append({
            'seed_seq': seed_seq,
            'sampled_seq': mutant_seq,
            'n_mutations': n_muts,
            'final_fitness': mutant_fitness,
            'delta_fitness': mutant_fitness - seed_fitness,
            'trajectory_idx': i,
            'method': 'random',
            'time': time_taken,
        })

    end_time = time.time()
    total_time = end_time - start_time
    time_per_sample = total_time / args.batch_size

    # Save
    df = pd.DataFrame(results)

    # Calculate additional sequence level properties
    print("\nComputing metrics...")
    all_sequences = df['sampled_seq'].tolist()
    oasis_humanness = compute_oasis_humanness(all_sequences)
    iglm_humanness = compute_iglm_humanness(all_sequences)
    edit_dist_from_seed = [calculate_hamming_distance(seq, seed_seq) for seq in all_sequences]
    df['oasis_humanness'] = oasis_humanness
    df['iglm_humanness'] = iglm_humanness
    df['edit_dist_from_seed'] = edit_dist_from_seed

    if args.output_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_path = f"random_{args.oracle}_seed{args.oracle_seed_idx}_mut{args.max_mutations}_{ts}.csv"

    if args.save_csv:
        df.to_csv(args.output_path, index=False)

    # Final Summary
    n_seqs = args.batch_size
    avg_fitness = df['final_fitness'].mean()
    max_fitness = df['final_fitness'].max()
    unique_seqs = df['sampled_seq'].nunique()
    int_div_mean, int_div_std = calculate_internal_diversity(all_sequences)

    print(f"\n{'='*60}")
    if args.save_csv:
        print(f"Results saved to: {args.output_path}")
    print(f"Final Population Size: {len(df)}")
    print(f"Unique Sequences: {unique_seqs} / {args.batch_size}")
    print(f"Proportion Unique: {unique_seqs / args.batch_size:.4f}")
    print(f"Avg Fitness: {avg_fitness:.4f} +/- {df['final_fitness'].std():.4f}")
    print(f"Max Fitness: {max_fitness:.4f}")
    print(f"Avg OASIS Humanness: {df['oasis_humanness'].mean():.4f} +/- {df['oasis_humanness'].std():.4f}")
    print(f"Avg IGLM Humanness: {df['iglm_humanness'].mean():.4f} +/- {df['iglm_humanness'].std():.4f}")
    print(f"Delta Fitness (Avg): {avg_fitness - seed_fitness:.4f}")
    print(f"Delta Fitness (Max): {max_fitness - seed_fitness:.4f}")
    print(f"Internal Diversity (Avg Hamming Dist): {int_div_mean:.4f} +/- {int_div_std:.4f}")
    print(f"Avg Mutations: {df['n_mutations'].mean():.4f} +/- {df['n_mutations'].std():.4f}")
    print(f"Total Runtime: {total_time:.4f}s")
    print(f"Avg Time per Sample: {time_per_sample:.4f}s")
    print(f"Total oracle calls: {oracle.get_call_count()}")
    print(f"Avg oracle calls per sample: {oracle.get_call_count() / n_seqs:.4f}")

if __name__ == "__main__":
    main()
