"""Oracle-guided evolutionary optimization using PEINT generative model.

Performs iterative genetic algorithm over multiple generations:
1. Sample sequences from current population using PEINT
2. Score sequences with oracle
3. Select top sequences for next generation
4. Track and visualize fitness improvement over generations
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from evo.oracles import get_oracle
from evo.tokenization import Vocab
from cosine.models.modules.peint_module import PEINTModule
from cosine.models.nets.peint import PEINTGenerator


def tokenize_sequence(sequence: str, vocab: Vocab, device: torch.device) -> torch.Tensor:
    """Tokenize a heavy chain sequence for PEINT.

    Since oracle seeds are heavy chain only, we'll use them as-is.
    PEINT was trained on paired sequences, so we need to handle single chains.
    """
    # Use vocab.encode which returns numpy array with BOS and EOS
    tokens = vocab.encode(sequence)  # Returns numpy array with BOS/EOS already added
    return torch.tensor(tokens, dtype=torch.long, device=device)


def decode_sequence(tokens: torch.Tensor, vocab: Vocab) -> str:
    """Decode tokenized sequence back to string."""
    # Convert to numpy if needed
    if torch.is_tensor(tokens):
        tokens = tokens.cpu().numpy()
    # Use vocab.decode which handles BOS/EOS automatically
    return vocab.decode(tokens)


def sample_children_from_parent(
    parent_seq: str,
    n_samples: int,
    generator: PEINTGenerator,
    vocab: Vocab,
    device: torch.device,
    branch_length_range: tuple = (1.0, 5.0),
    dummy_light_chain: str = "DIQMTQSPSSLSASVGDRVTITC",  # Short constant light chain
) -> list:
    """Sample child sequences from a parent using PEINT generator.

    Since PEINT was trained on paired sequences, we add a dummy light chain.
    We only return and evaluate the heavy chain part.

    Parameters
    ----------
    parent_seq : str
        Parent heavy chain sequence
    n_samples : int
        Number of children to sample
    generator : PEINTGenerator
        PEINT generator model
    vocab : Vocab
        Vocabulary for tokenization
    device : torch.device
        Device to run on
    branch_length_range : tuple
        Range of branch lengths to sample from (min, max)
    dummy_light_chain : str
        Constant light chain to pair with heavy chain

    Returns
    -------
    list
        List of sampled child heavy chain sequences (strings, heavy chain only)
    """
    # Create paired sequence with separator
    paired_parent = parent_seq + "." + dummy_light_chain

    # Tokenize paired sequence
    parent_tokens = tokenize_sequence(paired_parent, vocab, device)

    # Prepare batch: repeat parent n_samples times
    xs = parent_tokens.unsqueeze(0).repeat(n_samples, 1)

    # Sample branch lengths uniformly
    min_t, max_t = branch_length_range
    branch_lengths = torch.rand(n_samples, 1, device=device) * (max_t - min_t) + min_t
    ts = branch_lengths.repeat(1, 2)  # Same branch length for both chains

    # Set sizes for both chains
    # x_sizes should be shape (batch, seq_len) with sizes in first positions
    heavy_len = len(parent_seq) + 1  # +1 for BOS
    light_len = len(dummy_light_chain)  # Light chain doesn't have BOS (comes after separator)
    seq_len = len(parent_tokens)

    # Create x_sizes: (batch, seq_len) with sizes in first 2 positions
    x_sizes = torch.zeros(n_samples, seq_len, dtype=torch.long, device=device)
    x_sizes[:, 0] = heavy_len
    x_sizes[:, 1] = light_len

    # Create y_sizes: same format, +1 for separator in light chain
    y_sizes = torch.zeros(n_samples, seq_len, dtype=torch.long, device=device)
    y_sizes[:, 0] = heavy_len
    y_sizes[:, 1] = light_len + 1

    # Chain IDs: shape (batch, num_chains) - just [1, 2] for heavy and light
    chain_ids = torch.tensor([[1, 2]] * n_samples, dtype=torch.long, device=device)

    # Generate children
    with torch.no_grad():
        y_decoded = generator.dec_generate(
            ts=ts,
            xs=xs,
            x_sizes=x_sizes,
            y_sizes=y_sizes,
            chain_ids=chain_ids,
        )

    # Decode sequences and extract heavy chain only
    children = []
    for i in range(n_samples):
        child_paired = decode_sequence(y_decoded[i], vocab)
        # Extract heavy chain (before the separator ".")
        if "." in child_paired:
            child_heavy = child_paired.split(".")[0]
        else:
            child_heavy = child_paired  # Fallback if no separator
        children.append(child_heavy)

    return children


def evolutionary_optimization(
    oracle_name: str,
    n_generations: int,
    n_samples_per_parent: int,
    n_top_sequences: int,
    checkpoint_path: str,
    device: str = "cuda",
    branch_length_range: tuple = (1.0, 5.0),
    output_dir: str = "results/oracle_evolution",
    single_seed_start: bool = False,
    only_improvements: bool = False,
):
    """Run evolutionary optimization guided by oracle fitness.

    Parameters
    ----------
    oracle_name : str
        Name of oracle to use ('clone', 'rand_0.5', 'SARSCoV1', 'SARSCoV2Beta')
    n_generations : int
        Number of generations to evolve
    n_samples_per_parent : int
        Number of children to sample from each parent
    n_top_sequences : int
        Number of top sequences to keep each generation
    checkpoint_path : str
        Path to PEINT checkpoint
    device : str
        Device to use ('cuda' or 'cpu')
    branch_length_range : tuple
        Range of branch lengths for sampling
    output_dir : str
        Directory to save results
    single_seed_start : bool
        If True, start with only the best seed sequence duplicated n_top_sequences times
    only_improvements : bool
        If True, only keep sequences that improve upon the initial seed fitness
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load oracle
    print(f"\nLoading oracle: {oracle_name}")
    if oracle_name in ["SARSCoV1", "SARSCoV2Beta"]:
        oracle = get_oracle(oracle_name, device="cpu", use_iglm_weighting=False)
    else:
        oracle = get_oracle(oracle_name, device=device)

    print(f"  Chain type: {oracle.chain_type}")
    print(f"  Higher is better: {oracle.higher_is_better}")

    # Get seed sequences from oracle
    seed_sequences = oracle.seed_sequences
    print(f"  Number of seed sequences: {len(seed_sequences)}")
    print(f"  Seed sequence lengths: {[len(s) for s in seed_sequences]}")

    # Load PEINT model
    print(f"\nLoading PEINT checkpoint: {checkpoint_path}")
    module = PEINTModule.load_from_checkpoint(
        checkpoint_path,
        map_location=device,
        strict=True,
    )
    module = module.eval()
    vocab = module.net.vocab

    # Create generator
    generator = PEINTGenerator.from_peint(module.net).to(device)
    print(f"  Vocabulary size: {len(vocab)}")

    # Initialize population with seed sequences
    if single_seed_start:
        # Use only the best seed sequence, duplicated n_top_sequences times
        if oracle.higher_is_better:
            best_idx = np.argmax(oracle.seed_fitnesses)
        else:
            best_idx = np.argmin(oracle.seed_fitnesses)

        best_seed = seed_sequences[best_idx]
        best_fitness = oracle.seed_fitnesses[best_idx]

        current_population = [best_seed] * n_top_sequences
        current_fitnesses = [best_fitness] * n_top_sequences

        print(f"\nStarting with single best seed duplicated {n_top_sequences} times:")
        print(f"  Best seed fitness: {best_fitness:.6f}")
        print(f"  Seed sequence: {best_seed}")
    else:
        current_population = seed_sequences.copy()
        current_fitnesses = oracle.seed_fitnesses.copy()

    # Store initial seed fitness for filtering if needed
    initial_seed_fitness = (
        np.min(oracle.seed_fitnesses)
        if not oracle.higher_is_better
        else np.max(oracle.seed_fitnesses)
    )

    print(f"\n{'='*80}")
    print(f"Starting evolutionary optimization")
    print(f"{'='*80}")
    print(f"Oracle: {oracle_name}")
    print(f"Generations: {n_generations}")
    print(f"Samples per parent: {n_samples_per_parent}")
    print(f"Top sequences kept: {n_top_sequences}")
    print(f"Branch length range: {branch_length_range}")
    print(f"Single seed start: {single_seed_start}")
    print(f"Only improvements: {only_improvements}")
    if only_improvements:
        print(f"Initial seed fitness threshold: {initial_seed_fitness:.6f}")
    print(f"{'='*80}\n")

    # Track statistics over generations
    generation_stats = []
    all_sequences_history = []

    for gen in range(n_generations + 1):  # +1 to include generation 0 (seeds)
        print(f"\n{'='*80}")
        print(f"Generation {gen}")
        print(f"{'='*80}")

        if gen == 0:
            # Generation 0: just evaluate seeds
            print(f"Population size: {len(current_population)}")
            print(f"Fitness stats:")
            print(f"  Mean: {np.mean(current_fitnesses):.6f}")
            print(f"  Std:  {np.std(current_fitnesses):.6f}")
            print(
                f"  Best: {np.min(current_fitnesses) if not oracle.higher_is_better else np.max(current_fitnesses):.6f}"
            )

            # Record stats
            generation_stats.append(
                {
                    "generation": gen,
                    "mean_fitness": np.mean(current_fitnesses),
                    "std_fitness": np.std(current_fitnesses),
                    "best_fitness": (
                        np.min(current_fitnesses)
                        if not oracle.higher_is_better
                        else np.max(current_fitnesses)
                    ),
                    "worst_fitness": (
                        np.max(current_fitnesses)
                        if not oracle.higher_is_better
                        else np.min(current_fitnesses)
                    ),
                    "population_size": len(current_population),
                }
            )

            # Record sequences
            for seq, fit in zip(current_population, current_fitnesses):
                all_sequences_history.append(
                    {
                        "generation": gen,
                        "sequence": seq,
                        "fitness": fit,
                    }
                )

            continue

        # Sample children from each parent in current population
        print(
            f"Sampling {n_samples_per_parent} children from each of {len(current_population)} parents..."
        )
        all_children = []

        for parent_idx, parent_seq in enumerate(tqdm(current_population, desc="Sampling")):
            try:
                children = sample_children_from_parent(
                    parent_seq=parent_seq,
                    n_samples=n_samples_per_parent,
                    generator=generator,
                    vocab=vocab,
                    device=device,
                    branch_length_range=branch_length_range,
                )
                all_children.extend(children)
            except Exception as e:
                print(f"  Warning: Failed to sample from parent {parent_idx}: {e}")
                continue

        print(f"  Sampled {len(all_children)} total children")

        # Combine parents and children
        candidate_sequences = current_population + all_children
        print(f"  Total candidates (parents + children): {len(candidate_sequences)}")

        # Score all candidates with oracle
        print(f"Scoring candidates with oracle...")
        candidate_fitnesses = oracle.predict_batch(candidate_sequences)

        # Filter for only improvements if requested
        if only_improvements:
            if oracle.higher_is_better:
                # Keep only sequences better than initial seed
                improvement_mask = np.array(candidate_fitnesses) > initial_seed_fitness
            else:
                # Keep only sequences better than initial seed (lower is better)
                improvement_mask = np.array(candidate_fitnesses) < initial_seed_fitness

            filtered_sequences = [
                seq for seq, keep in zip(candidate_sequences, improvement_mask) if keep
            ]
            filtered_fitnesses = [
                fit for fit, keep in zip(candidate_fitnesses, improvement_mask) if keep
            ]

            print(
                f"  Improvements only: {len(filtered_sequences)}/{len(candidate_sequences)} sequences improve upon initial seed"
            )

            if len(filtered_sequences) == 0:
                print(f"  Warning: No improvements found, keeping all candidates")
                filtered_sequences = candidate_sequences
                filtered_fitnesses = candidate_fitnesses
        else:
            filtered_sequences = candidate_sequences
            filtered_fitnesses = candidate_fitnesses

        # Select top sequences from filtered pool
        if oracle.higher_is_better:
            top_indices = np.argsort(filtered_fitnesses)[-n_top_sequences:][::-1]
        else:
            top_indices = np.argsort(filtered_fitnesses)[:n_top_sequences]

        # Handle case where we have fewer improvements than n_top_sequences
        if len(filtered_sequences) < n_top_sequences:
            print(f"  Warning: Only {len(filtered_sequences)} sequences available, keeping all")
            current_population = filtered_sequences
            current_fitnesses = filtered_fitnesses
        else:
            current_population = [filtered_sequences[i] for i in top_indices]
            current_fitnesses = [filtered_fitnesses[i] for i in top_indices]

        print(f"\nGeneration {gen} results:")
        print(f"  Population size: {len(current_population)}")
        print(f"  Fitness stats:")
        print(f"    Mean: {np.mean(current_fitnesses):.6f}")
        print(f"    Std:  {np.std(current_fitnesses):.6f}")
        print(
            f"    Best: {np.min(current_fitnesses) if not oracle.higher_is_better else np.max(current_fitnesses):.6f}"
        )
        print(
            f"    Worst: {np.max(current_fitnesses) if not oracle.higher_is_better else np.min(current_fitnesses):.6f}"
        )

        # Record stats
        generation_stats.append(
            {
                "generation": gen,
                "mean_fitness": np.mean(current_fitnesses),
                "std_fitness": np.std(current_fitnesses),
                "best_fitness": (
                    np.min(current_fitnesses)
                    if not oracle.higher_is_better
                    else np.max(current_fitnesses)
                ),
                "worst_fitness": (
                    np.max(current_fitnesses)
                    if not oracle.higher_is_better
                    else np.min(current_fitnesses)
                ),
                "population_size": len(current_population),
            }
        )

        # Record all candidates
        for seq, fit in zip(candidate_sequences, candidate_fitnesses):
            all_sequences_history.append(
                {
                    "generation": gen,
                    "sequence": seq,
                    "fitness": fit,
                }
            )

    # Save results
    print(f"\n{'='*80}")
    print(f"Saving results to {output_path}")
    print(f"{'='*80}")

    # Save generation statistics
    stats_df = pd.DataFrame(generation_stats)
    stats_path = output_path / f"{oracle_name}_generation_stats.csv"
    stats_df.to_csv(stats_path, index=False)
    print(f"Saved generation statistics: {stats_path}")

    # Save all sequences
    sequences_df = pd.DataFrame(all_sequences_history)
    sequences_path = output_path / f"{oracle_name}_all_sequences.csv"
    sequences_df.to_csv(sequences_path, index=False)
    print(f"Saved all sequences: {sequences_path}")

    # Save final population
    final_pop_df = pd.DataFrame(
        {
            "sequence": current_population,
            "fitness": current_fitnesses,
        }
    )
    final_pop_path = output_path / f"{oracle_name}_final_population.csv"
    final_pop_df.to_csv(final_pop_path, index=False)
    print(f"Saved final population: {final_pop_path}")

    # Plot results
    print(f"\nGenerating plots...")

    # Plot fitness over generations
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Mean fitness with std
    ax = axes[0]
    ax.plot(stats_df["generation"], stats_df["mean_fitness"], "o-", label="Mean", linewidth=2)
    ax.fill_between(
        stats_df["generation"],
        stats_df["mean_fitness"] - stats_df["std_fitness"],
        stats_df["mean_fitness"] + stats_df["std_fitness"],
        alpha=0.3,
    )
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    ax.set_title(f"Mean Fitness over Generations\n({oracle_name})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Best and worst fitness
    ax = axes[1]
    ax.plot(stats_df["generation"], stats_df["best_fitness"], "o-", label="Best", linewidth=2)
    ax.plot(stats_df["generation"], stats_df["worst_fitness"], "s-", label="Worst", linewidth=2)
    ax.plot(stats_df["generation"], stats_df["mean_fitness"], "^-", label="Mean", linewidth=2)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    ax.set_title(f"Best/Worst/Mean Fitness over Generations\n({oracle_name})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = output_path / f"{oracle_name}_fitness_evolution.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot: {plot_path}")
    plt.close()

    # Print final summary
    print(f"\n{'='*80}")
    print(f"FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"Oracle: {oracle_name}")
    print(f"Generations completed: {n_generations}")
    print(f"\nFitness improvement:")
    print(f"  Initial (Gen 0):")
    print(f"    Mean: {generation_stats[0]['mean_fitness']:.6f}")
    print(f"    Best: {generation_stats[0]['best_fitness']:.6f}")
    print(f"  Final (Gen {n_generations}):")
    print(f"    Mean: {generation_stats[-1]['mean_fitness']:.6f}")
    print(f"    Best: {generation_stats[-1]['best_fitness']:.6f}")

    if oracle.higher_is_better:
        improvement = generation_stats[-1]["best_fitness"] - generation_stats[0]["best_fitness"]
        print(f"  Improvement: +{improvement:.6f}")
    else:
        improvement = generation_stats[0]["best_fitness"] - generation_stats[-1]["best_fitness"]
        print(f"  Improvement: {improvement:.6f} (lower is better)")

    print(f"\nBest sequence found:")
    best_idx = (
        np.argmax(current_fitnesses) if oracle.higher_is_better else np.argmin(current_fitnesses)
    )
    print(f"  Fitness: {current_fitnesses[best_idx]:.6f}")
    print(f"  Sequence: {current_population[best_idx]}")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Oracle-guided evolutionary optimization using PEINT"
    )
    parser.add_argument(
        "--oracle",
        type=str,
        default="clone",
        choices=["clone", "rand_0.0", "rand_0.5", "rand_1.0", "SARSCoV1", "SARSCoV2Beta"],
        help="Oracle to use for fitness evaluation",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=5,
        help="Number of generations to evolve",
    )
    parser.add_argument(
        "--samples-per-parent",
        type=int,
        default=10,
        help="Number of children to sample from each parent",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top sequences to keep each generation",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/accounts/projects/yss/stephen.lu/peint/logs/train/runs/2025-11-01_03-40-52/checkpoints/epoch_031.ckpt",
        help="Path to PEINT checkpoint",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda or cpu)",
    )
    parser.add_argument(
        "--branch-length-min",
        type=float,
        default=1.0,
        help="Minimum branch length for sampling",
    )
    parser.add_argument(
        "--branch-length-max",
        type=float,
        default=5.0,
        help="Maximum branch length for sampling",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/oracle_evolution",
        help="Directory to save results",
    )
    parser.add_argument(
        "--single-seed-start",
        action="store_true",
        help="Start with only the best seed sequence duplicated (default: use all seeds)",
    )
    parser.add_argument(
        "--only-improvements",
        action="store_true",
        help="Only keep sequences that improve upon the initial seed fitness",
    )

    args = parser.parse_args()

    evolutionary_optimization(
        oracle_name=args.oracle,
        n_generations=args.generations,
        n_samples_per_parent=args.samples_per_parent,
        n_top_sequences=args.top_k,
        checkpoint_path=args.checkpoint,
        device=args.device,
        branch_length_range=(args.branch_length_min, args.branch_length_max),
        output_dir=args.output_dir,
        single_seed_start=args.single_seed_start,
        only_improvements=args.only_improvements,
    )


if __name__ == "__main__":
    main()
