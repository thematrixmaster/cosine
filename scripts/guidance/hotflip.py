"""
Stochastic HotFlip Baseline (Mutation-based Hill Climbing)

Iteratively improves a sequence by sampling mutations from the Top-K candidates.
Supports two modes:
1. Brute Force (--no-gradients): Evaluates all L*20 mutations exactly. Slow but accurate.
2. Gradient Approximation (--use-gradients): Estimates gains via 1 backward pass. Fast but approximate.

Usage:
    # Accurate, slower (Brute Force)
    uv run python scripts/baseline_hotflip.py \
        --oracle SARSCoV2Beta --oracle-seed-idx 0 \
        --max-mutations 20 --top-k 10

    # Fast, approximate (Gradient)
    uv run python scripts/baseline_hotflip.py \
        --oracle SARSCoV2Beta --oracle-seed-idx 0 \
        --max-mutations 20 --top-k 10 \
        --use-gradients
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

# Standard amino acid vocabulary (must match model's order)
AA_VOCAB = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_VOCAB)}

def parse_args():
    parser = argparse.ArgumentParser(description='Stochastic HotFlip Baseline')
    
    # Oracle & Seed
    parser.add_argument('--oracle', type=str, default='SARSCoV2Beta', choices=['SARSCoV1', 'SARSCoV2Beta'])
    parser.add_argument('--oracle-seed-idx', type=int, default=0)
    
    # Sampling Params
    parser.add_argument('--batch-size', type=int, default=100, help='Number of independent trajectories')
    parser.add_argument('--max-mutations', type=int, default=10, help='Maximum mutations allowed per trajectory')
    parser.add_argument('--top-k', type=int, default=10, help='Sample from top K best mutations')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature for energy distribution')
    parser.add_argument('--min-improvement', type=float, default=1e-4, help='Stop if gain is below this threshold')
    
    # Method Selection
    parser.add_argument('--use-gradients', action='store_true', 
                        help='Use gradient approximation (fast) instead of brute force (slow)')
    
    # Constraints
    parser.add_argument('--mask-region', type=str, default=None, 
                        choices=['CDR1', 'CDR2', 'CDR3', 'CDR_overall', 'FR1', 'FR2', 'FR3', 'FR4', None])
    
    # Tech
    parser.add_argument('--save-csv', action='store_true', help='Whether to save output as CSV')
    parser.add_argument('--oracle-chunk-size', type=int, default=2000, help='Batch size for brute-force inference')
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

# --- Calculation Strategies ---

def get_gains_brute_force(current_seq, current_score, mask, oracle, args):
    """
    Exact calculation:
    1. Generate all strings
    2. Run forward pass on all of them
    3. Return (candidate_sequences, candidate_gains, candidate_scores)
    """
    mutants = []
    mutant_indices = [] # (pos, aa_idx) to reconstruct later if needed
    
    seq_list = list(current_seq)
    
    # Generate all single mutants
    for i, char in enumerate(seq_list):
        if not mask[i]: continue
        for aa_idx, aa in enumerate(AA_VOCAB):
            if aa == char: continue
            
            mut_list = seq_list.copy()
            mut_list[i] = aa
            mutants.append("".join(mut_list))

    if not mutants:
        return [], [], []

    # Batch inference
    scores = oracle.predict_batch(mutants)[0]
    scores = np.array(scores)
    gains = scores - current_score
    
    return mutants, gains, scores


def get_gains_gradients(current_seq, current_score, mask, oracle, args):
    """
    Approximate calculation:
    1. Run 1 backward pass
    2. Extract gradients as proxies for fitness gains
    3. Return (candidate_sequences, candidate_gains, candidate_scores)
    """
    # oracle.compute_fitness_gradient returns (L, Vocab)
    grads, _ = oracle.compute_fitness_gradient(current_seq)
    
    # Handle potential batch dimension
    if grads.ndim == 3:
        grads = grads.squeeze(0)

    # Apply Mask (Set frozen regions to -inf)
    grads[~mask, :] = -float('inf')

    # Mask self-transitions (diagonal) to -inf
    # Also: Prepare to capture the gradient of the CURRENT sequence for subtraction
    current_seq_grads = np.zeros(len(current_seq))
    
    seq_indices = [AA_TO_IDX.get(c, -1) for c in current_seq]
    
    for i, aa_idx in enumerate(seq_indices):
        if aa_idx != -1 and aa_idx < grads.shape[1]:
            # 1. Store the gradient of the character we are about to lose
            current_seq_grads[i] = grads[i, aa_idx]
            
            # 2. Mask self-transition
            grads[i, aa_idx] = -float('inf')
        else:
            # If current char is unknown/special, assume 0 contribution to gradient math
            current_seq_grads[i] = 0.0

    # Flatten to find top candidates
    flat_grads = grads.flatten()
    
    # Filter out -inf
    valid_indices = np.where(flat_grads > -1e9)[0]
    
    if len(valid_indices) == 0:
        return [], [], []

    candidates = []
    candidate_gains = []
    candidate_scores = [] 
    
    for idx in valid_indices:
        # Get position and new AA index
        pos, aa_idx = np.unravel_index(idx, grads.shape)
        
        if aa_idx >= len(AA_VOCAB):
            continue
            
        aa = AA_VOCAB[aa_idx]
        
        # --- THE FIX IS HERE ---
        # Gain = Gradient(New) - Gradient(Old)
        grad_new = flat_grads[idx]
        grad_old = current_seq_grads[pos]
        
        # Calculate Net Gain
        gain = grad_new - grad_old
        
        seq_list = list(current_seq)
        seq_list[pos] = aa
        
        candidates.append("".join(seq_list))
        candidate_gains.append(gain)
        
        # Approximate new score
        candidate_scores.append(current_score + gain)
        
    return candidates, np.array(candidate_gains), np.array(candidate_scores)

# --- Core Logic ---

def run_hotflip_trajectory(
    start_seq: str,
    oracle,
    args,
    mask: np.ndarray,
    device
) -> tuple[str, int, float]:
    
    current_seq = start_seq
    # Initial score
    current_score = oracle.predict(start_seq)[0]
    traj_mutations = 0
    
    for step in range(args.max_mutations):
        
        # 1. Get Gains (Strategy Pattern)
        if args.use_gradients:
            candidates, gains, scores = get_gains_gradients(current_seq, current_score, mask, oracle, args)
        else:
            candidates, gains, scores = get_gains_brute_force(current_seq, current_score, mask, oracle, args)
            
        if len(candidates) == 0:
            break
            
        # 2. Select Top-K
        k = min(args.top_k, len(candidates))
        
        # argpartition is faster than sort for top-k
        top_k_indices = np.argpartition(gains, -k)[-k:]
        
        top_k_gains = gains[top_k_indices]
        top_k_candidates = [candidates[i] for i in top_k_indices]
        top_k_scores = scores[top_k_indices]
        
        # 3. Check Termination (Best possible gain < threshold)
        best_gain = np.max(top_k_gains)
        if best_gain < args.min_improvement:
            # print(f"Stopping: Gain {best_gain:.5f} < {args.min_improvement}")
            break 
            
        # 4. Energy-based Sampling
        # P(x) = exp(gain / T) / Z  (Using gain is equivalent to score for relative probability)
        logits = top_k_gains / args.temperature
        probs = np.exp(logits - np.max(logits)) # Shift for stability
        probs = probs / probs.sum()
        
        # Sample
        chosen_local_idx = np.random.choice(len(top_k_candidates), p=probs)
        next_seq = top_k_candidates[chosen_local_idx]
        
        # 5. Update State
        current_seq = next_seq
        
        # If we used gradients, the 'score' was approximate. 
        # We should compute the TRUE score now for accurate logging/next step.
        if args.use_gradients:
            current_score = oracle.predict(current_seq)[0]
        else:
            current_score = top_k_scores[chosen_local_idx]
            
        traj_mutations += 1
        
    return current_seq, traj_mutations, current_score

# --- Main ---

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Reproducibility
    seed = args.seed if args.seed is not None else int(time.time())
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    print(f"{'='*60}\nRunning Stochastic HotFlip Baseline\n{'='*60}")
    print(f"Mode: {'Gradient Approximation' if args.use_gradients else 'Brute Force (Exact)'}")

    # Load Oracle
    print(f"Loading Oracle: {args.oracle}")
    oracle = OracleCallCounter(get_oracle(
        args.oracle,
        enable_mc_dropout=True,
        mc_samples=10,
        mc_dropout_seed=42,
        use_iglm_weighting=False,
        device=device,
        precompute_seed_fitnesses=False,
    ))

    # Load Seed
    seed_seq = select_oracle_seed(args.oracle, args.oracle_seed_idx)
    seed_fitness = oracle.predict(seed_seq, increment=False)[0]
    print(f"Seed Fitness: {seed_fitness}")

    # Create Mask
    mask = get_region_mask(seed_seq, args.mask_region)
    if args.mask_region:
        print(f"Constraining mutations to {args.mask_region} ({mask.sum()} residues)")

    results = []

    print(f"\nStarting {args.batch_size} trajectories...")

    start_time = time.time()
    for i in tqdm(range(args.batch_size)):
        traj_start_time = time.time()
        final_seq, n_muts, final_score = run_hotflip_trajectory(
            start_seq=seed_seq,
            oracle=oracle,
            args=args,
            mask=mask,
            device=device
        )
        traj_end_time = time.time()
        time_taken = traj_end_time - traj_start_time
        results.append({
            'seed_seq': seed_seq,
            'sampled_seq': final_seq,
            'n_mutations': n_muts,
            'final_fitness': final_score,
            'delta_fitness': final_score - seed_fitness,
            'trajectory_idx': i,
            'method': 'gradient' if args.use_gradients else 'brute_force',
            'time': time_taken,
        })
    end_time = time.time()
    total_time = end_time - start_time
    time_per_sample = total_time / args.batch_size

    # Save
    df = pd.DataFrame(results)

    # Calculate additional sequence level properties like oasis humanness and edit distance from the seed
    all_sequences = df['sampled_seq'].tolist()
    oasis_humanness = compute_oasis_humanness(all_sequences)
    iglm_humanness = compute_iglm_humanness(all_sequences)
    edit_dist_from_seed = [calculate_hamming_distance(seq, seed_seq) for seq in all_sequences]
    df['oasis_humanness'] = oasis_humanness
    df['iglm_humanness'] = iglm_humanness
    df['edit_dist_from_seed'] = edit_dist_from_seed
    
    
    if args.output_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        method_str = "grad" if args.use_gradients else "exact"
        args.output_path = f"hotflip_{method_str}_{args.oracle}_seed{args.oracle_seed_idx}_k{args.top_k}_{ts}.csv"

    if args.save_csv:
        df.to_csv(args.output_path, index=False)

    # Final Summary
    print("\nComputing metrics...")
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