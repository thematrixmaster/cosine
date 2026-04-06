"""
Genetic Algorithm (Directed Evolution) Baseline

Simulates a standard directed evolution experiment:
1. Starts with a population of the seed sequence.
2. In each generation, selects the top-K variants (Truncation Selection).
3. Repopulates the next generation by randomly mutating the survivors.
4. Returns the final population as the set of sampled sequences.

Outputs a CSV matching the format of the HotFlip baseline for easy comparison.

Usage:
    uv run python scripts/genetic.py \
        --oracle SARSCoV2Beta \
        --oracle-seed-idx 0 \
        --pop-size 100 \
        --top-k 50 \
        --generations 20 \
        --max-mutations 5 \
        --mutation-rate 1
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
    parser = argparse.ArgumentParser(description='Genetic Algorithm Baseline')
    
    # Oracle & Seed
    parser.add_argument('--oracle', type=str, default='SARSCoV2Beta', choices=['SARSCoV1', 'SARSCoV2Beta'])
    parser.add_argument('--oracle-seed-idx', type=int, default=0)
    parser.add_argument('--seed-seq', type=str, default=None,
                        help='Custom seed sequence to optimize (overrides oracle-seed-idx if provided)')
    
    # GA Hyperparameters
    parser.add_argument('--pop-size', type=int, default=100, help='Population size (acts as batch size)')
    parser.add_argument('--top-k', type=int, default=10, help='Number of parents selected for the next generation')
    parser.add_argument('--generations', type=int, default=20, help='Number of evolution rounds')
    parser.add_argument('--mutation-rate', type=int, default=1, help='Number of mutations to introduce per child')
    
    # Constraints
    parser.add_argument('--max-mutations', type=int, default=10, help='Max mutations allowed relative to seed')
    parser.add_argument('--mask-region', type=str, default=None, 
                        choices=['CDR1', 'CDR2', 'CDR3', 'CDR_overall', 'FR1', 'FR2', 'FR3', 'FR4', None])
    
    # Tech
    parser.add_argument('--save-csv', action='store_true', help='Whether to save output as CSV')
    parser.add_argument('--oracle-chunk-size', type=int, default=2000, help='Batch size for oracle inference')
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
    # Adjust path if needed to match your environment
    csv_path = Path("/accounts/projects/yss/stephen.lu/peint-workspace/main/evo/evo/oracles/data") / csv_files[oracle_variant]
    
    # Fallback if hardcoded path fails
    if not csv_path.exists():
        csv_path = Path(__file__).parent.parent / "evo/evo/oracles/data" / csv_files[oracle_variant]

    df = pd.read_csv(csv_path)
    df.columns = ['sequence', 'binding']
    return df[df['binding'] == 1].reset_index(drop=True)

def select_oracle_seed(oracle_variant: str, seed_idx: int) -> str:
    binders = load_oracle_binders(oracle_variant)
    return binders.iloc[seed_idx]['sequence']

def get_region_mask(sequence: str, region_name: str) -> np.ndarray:
    if region_name is None:
        return np.ones(len(sequence), dtype=bool)
    masks = create_region_masks(sequence, scheme="imgt")
    return masks[region_name]

def mutate_boundary_aware(parent_seq: str, seed_seq: str, mask: np.ndarray, max_muts: int) -> str:
    """
    Smart mutation that restricts choices based on distance to seed.
    Guarantees a valid child (dist <= max_muts) so no rejection loop is needed.
    """
    seq_list = list(parent_seq)
    
    # Identify indices allowed by the region mask
    region_indices = np.where(mask)[0]
    
    if len(region_indices) == 0:
        return parent_seq

    # Identify which indices are currently mutated relative to seed
    # (Only consider mutations within the masked region)
    current_mutations = [i for i in region_indices if parent_seq[i] != seed_seq[i]]
    current_dist = len(current_mutations)
    
    # LOGIC:
    # 1. If we are BELOW the limit, we can mutate ANY valid region position.
    if current_dist < max_muts:
        candidates = region_indices
        
    # 2. If we are AT the limit, we can ONLY mutate positions that are ALREADY mutated.
    #    - Changing a mutated residue to a NEW amino acid -> Distance stays same (Lateral).
    #    - Changing a mutated residue back to Seed AA -> Distance goes down (Reversion).
    else:
        candidates = current_mutations
        # Edge case: If at max_muts but no candidates found (should be impossible if max_muts > 0)
        if not candidates: 
            candidates = region_indices

    # Pick a position to mutate
    idx = random.choice(candidates)
    
    # Pick a new AA
    current_aa = seq_list[idx]
    possible_aas = [aa for aa in AA_VOCAB if aa != current_aa]
    new_aa = random.choice(possible_aas)
    
    seq_list[idx] = new_aa
    return "".join(seq_list)

# --- Core GA Logic ---

def run_genetic_algorithm(
    seed_seq: str,
    oracle,
    args,
    mask: np.ndarray,
) -> list[tuple]:
    """
    Returns: List of (sequence, score) for the final population.
    """
    
    # 1. Initialize Population (All copies of seed)
    population = [seed_seq] * args.pop_size
    score_cache = {}
    
    print(f"Starting Evolution: Pop {args.pop_size} | Gens {args.generations} | Top-K {args.top_k}")

    for gen in tqdm(range(args.generations), desc="Evolving"):
        
        # 2. Evaluate Fitness
        # Filter uncached sequences
        to_evaluate = [seq for seq in population if seq not in score_cache]
        
        if to_evaluate:
            # Batch inference
            scores = oracle.predict_batch(to_evaluate)[0]
            for seq, score in zip(to_evaluate, scores):
                score_cache[seq] = score
        
        # Create (seq, score) pairs
        pop_scores = [(seq, score_cache[seq]) for seq in population]
        
        # 3. Selection (Truncation)
        # Sort descending by fitness
        pop_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Elites (Top K)
        parents = [p[0] for p in pop_scores[:args.top_k]]
        
        # 4. Repopulate (Mutation)
        # Elites survive automatically
        next_gen = list(parents)
        
        # Fill the rest of the population
        while len(next_gen) < args.pop_size:
            parent = random.choice(parents)
            
            # Use Boundary-Aware Mutation
            # This logic guarantees the child is valid, so we don't need the if/else reject loop
            child = mutate_boundary_aware(parent, seed_seq, mask, args.max_mutations)
            next_gen.append(child)
                
        population = next_gen
        
    # Final Evaluation for the last generation
    final_population_scores = []
    
    # Ensure all final seqs are scored
    to_evaluate = [seq for seq in population if seq not in score_cache]
    if to_evaluate:
        scores = oracle.predict_batch(to_evaluate, increment=False)[0]
        for seq, score in zip(to_evaluate, scores):
            score_cache[seq] = score
            
    for seq in population:
        final_population_scores.append((seq, score_cache[seq]))
        
    return final_population_scores

# --- Main ---

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Reproducibility
    seed = args.seed if args.seed is not None else int(time.time())
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    print(f"{'='*60}\nRunning Genetic Algorithm Baseline\n{'='*60}")
    
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
    print(f"Seed Fitness: {seed_fitness}")
    
    # Create Mask
    mask = get_region_mask(seed_seq, args.mask_region)
    if args.mask_region:
        print(f"Constraining mutations to {args.mask_region} ({mask.sum()} residues)")

    # Run GA and time it
    start_time = time.time()
    final_pop = run_genetic_algorithm(seed_seq, oracle, args, mask)
    end_time = time.time()
    
    total_time = end_time - start_time
    # Amortize time per sample for the dataframe column
    time_per_sample = total_time / args.pop_size
    
    # Prepare Results Dataframe
    results = []
    
    for idx, (sampled_seq, final_fitness) in enumerate(final_pop):
        n_muts = calculate_hamming_distance(sampled_seq, seed_seq)
        
        results.append({
            'seed_seq': seed_seq,
            'sampled_seq': sampled_seq,
            'n_mutations': n_muts,
            'final_fitness': final_fitness,
            'delta_fitness': final_fitness - seed_fitness,
            'trajectory_idx': idx,
            'method': 'genetic_algorithm',
            'time': time_per_sample, # Amortized time
        })

    df = pd.DataFrame(results)
    
    # Compute Humanness and Diversity Metrics
    print("\nComputing metrics...")
    all_sequences = df['sampled_seq'].tolist()
    
    df['oasis_humanness'] = compute_oasis_humanness(all_sequences)
    df['iglm_humanness'] = compute_iglm_humanness(all_sequences)
    df['edit_dist_from_seed'] = df['n_mutations'] # Redundant but requested explicitly
    
    # Save
    if args.output_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_path = f"ga_{args.oracle}_seed{args.oracle_seed_idx}_pop{args.pop_size}_gen{args.generations}_{ts}.csv"

    if args.save_csv:
        df.to_csv(args.output_path, index=False)
    
    # Final Summary
    n_seqs = args.pop_size
    avg_fitness = df['final_fitness'].mean()
    max_fitness = df['final_fitness'].max()
    unique_seqs = df['sampled_seq'].nunique()
    int_div_mean, int_div_std = calculate_internal_diversity(all_sequences)

    print(f"\n{'='*60}")
    if args.save_csv:
        print(f"Results saved to: {args.output_path}")
    print(f"Final Population Size: {len(df)}")
    print(f"Unique Sequences: {unique_seqs} / {args.pop_size}")
    print(f"Proportion Unique: {unique_seqs / args.pop_size:.4f}")
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