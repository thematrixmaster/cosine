"""Compare guided vs unguided CTMC sampling with oracle scoring.

Creates ICML-style publication-ready plots.
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import time
import random
from datetime import datetime

from evo.oracles import get_oracle, CovidOracle
from evo.tokenization import Vocab
from cosine.models.modules.ctmc_module import CTMCModule
from cosine.models.nets.ctmc import NeuralCTMC, NeuralCTMCGenerator


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compare guided vs unguided CTMC sampling'
    )
    parser.add_argument('--methods', nargs='+',
                        choices=['unguided', 'exact_guided', 'taylor_guided'],
                        default=['unguided', 'taylor_guided'],
                        help='Methods to compare (default: unguided taylor_guided)')
    parser.add_argument('--branch-length', type=float, default=0.5,
                        help='Branch length for sampling (default: 0.5)')
    parser.add_argument('--num-samples', type=int, default=50,
                        help='Number of samples per method (default: 50)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size for sampling (default: same as num-samples)')
    parser.add_argument('--guidance-strength', type=float, default=2.0,
                        help='Guidance strength γ for guided methods (default: 2.0)')
    parser.add_argument('--output-dir', type=str,
                        default='/scratch/users/stephen.lu/projects/protevo/results/guided_vs_unguided',
                        help='Output directory for results')
    parser.add_argument('--model-path', type=str,
                        default='/scratch/users/stephen.lu/projects/protevo/logs/train/runs/2026-01-06_18-32-49/checkpoints/epoch_001.ckpt',
                        help='Path to CTMC model checkpoint')
    parser.add_argument('--oracle-chunk-size', type=int, default=5000,
                        help='Max sequences per oracle call to avoid OOM (default: 5000)')
    parser.add_argument('--num-seeds', type=int, default=None,
                        help='Number of seed sequences to use (default: all available seeds)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility (default: None, uses timestamp)')
    parser.add_argument('--variant', type=str, default='SARSCoV1',
                        choices=['SARSCoV1', 'SARSCoV2Beta'],
                        help='Oracle variant to use (default: SARSCoV1)')
    return parser.parse_args()


def main():
    args = parse_args()

    # Set random seed for reproducibility
    if args.seed is not None:
        random_seed = args.seed
    else:
        random_seed = int(time.time())

    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    # Parameters from args
    methods = args.methods
    branch_length = args.branch_length
    num_samples = args.num_samples
    batch_size = args.batch_size if args.batch_size is not None else num_samples
    guidance_strength = args.guidance_strength
    oracle_chunk_size = args.oracle_chunk_size
    variant = args.variant
    num_seeds = args.num_seeds
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create experiment-specific subdirectory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    method_str = "_".join([m.replace("_guided", "") for m in methods])
    bl_str = f"{branch_length:.2f}".replace(".", "p")
    exp_name = f"{timestamp}_{variant}_{method_str}_bl{bl_str}_n{num_samples}_seed{random_seed}"
    if num_seeds is not None:
        exp_name += f"_nseeds{num_seeds}"

    exp_dir = output_dir / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    print(f"Experiment directory: {exp_dir}")
    print(f"Random seed: {random_seed}\n")

    # Load model
    model_path = Path(args.model_path)
    print(f"Loading CTMC model from: {model_path}")
    module = CTMCModule.load_from_checkpoint(str(model_path), map_location=device, strict=False)
    net: NeuralCTMC = module.net
    vocab: Vocab = net.vocab
    net = net.eval().to(device)
    generator = NeuralCTMCGenerator(neural_ctmc=net)

    # Load oracle on CUDA with MC Dropout
    print(f"Loading oracle ({variant}) on {device} with MC Dropout...")
    oracle: CovidOracle = get_oracle(
        variant,
        enable_mc_dropout=True,
        mc_samples=10,
        use_iglm_weighting=False,
        device=device,
    )

    # Get all seed sequences
    all_seed_sequences = list(oracle.seed_data.values())
    total_available_seeds = len(all_seed_sequences)

    # Select subset of seeds if requested
    if num_seeds is not None:
        if num_seeds > total_available_seeds:
            print(f"Warning: Requested {num_seeds} seeds but only {total_available_seeds} available. Using all.")
            num_seeds = total_available_seeds
        seed_indices = random.sample(range(total_available_seeds), num_seeds)
        seed_sequences = [all_seed_sequences[i] for i in sorted(seed_indices)]
        print(f"Selected {num_seeds} seeds from {total_available_seeds} available")
    else:
        seed_sequences = all_seed_sequences
        print(f"Using all {total_available_seeds} available seed sequences")

    print(f"\n{'='*80}")
    print("GUIDED VS UNGUIDED CTMC SAMPLING COMPARISON")
    print(f"{'='*80}")
    print(f"Variant: {variant}")
    print(f"Branch length: {branch_length}")
    print(f"Methods: {', '.join(methods)}")
    print(f"Samples per method: {num_samples}")
    print(f"Guidance strength (γ): {guidance_strength}")
    print(f"Random seed: {random_seed}")
    print(f"Number of seed sequences: {len(seed_sequences)}")
    print(f"Oracle: {variant} neutralization with MC Dropout (10 samples)")
    print(f"\nTotal trajectories: {len(methods) * num_samples * len(seed_sequences)}")
    print(f"  {len(methods)} methods × {num_samples} samples × {len(seed_sequences)} seeds")
    print(f"{'='*80}\n")

    all_results = []
    runtime_data = []
    total_start_time = time.time()

    # Helper function to count mutations
    def count_mutations(seq, seed):
        min_len = min(len(seq), len(seed))
        return sum(1 for i in range(min_len) if seq[i] != seed[i])

    # Loop over seeds
    for seed_idx, seed_data in enumerate(seed_sequences):
        seed_seq_str = seed_data["sequence"]
        seed_fitness = seed_data["fitness"]

        print(f"\n{'='*80}")
        print(f"Seed {seed_idx}: {seed_seq_str[:40]}...")
        print(f"Seed fitness: {seed_fitness:.4f} | Sequence length: {len(seed_seq_str)}")
        print(f"{'='*80}")

        # Convert seed to tensor
        x = torch.tensor([vocab.tokens_to_idx[aa] for aa in seed_seq_str], device=device).unsqueeze(0)
        x_sizes = torch.tensor([len(seed_seq_str)], device=device)
        t = torch.tensor([branch_length], device=device)

        # Sample for each method
        for method in methods:
            # Determine sampling parameters
            if method == 'unguided':
                use_guided = False
                use_taylor = False
                batch_size = args.batch_size
            elif method == 'exact_guided':
                use_guided = True
                use_taylor = False
                batch_size = 5 # use smaller batch size for exact guided since it's slow
            elif method == 'taylor_guided':
                use_guided = True
                use_taylor = True
                batch_size = args.batch_size

            print(f"\n[{method.upper()}] Sampling {num_samples} sequences (batch_size={batch_size})...")
            start_time = time.time()
            sampled_seqs = []

            # Process in batches
            for batch_start in range(0, num_samples, batch_size):
                print(f"Processing batch {batch_start} to {batch_start + batch_size}...")
                batch_end = min(batch_start + batch_size, num_samples)
                current_batch_size = batch_end - batch_start

                # Replicate seed sequence for this batch
                x_batch = x.repeat(current_batch_size, 1)
                t_batch = t.repeat(current_batch_size)
                x_sizes_batch = x_sizes.repeat(current_batch_size)

                with (torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16)):
                    if use_guided:
                        y_batch = generator.generate_with_gillespie(
                            x=x_batch,
                            t=t_batch,
                            x_sizes=x_sizes_batch,
                            oracle=oracle,
                            guidance_strength=guidance_strength,
                            use_guidance=True,
                            temperature=1.0,
                            no_special_toks=True,
                            max_decode_steps=1000,
                            use_scalar_steps=False,
                            use_taylor_approx=use_taylor,
                            verbose=(batch_start == 0),
                            oracle_chunk_size=oracle_chunk_size,
                        )
                    else:
                        y_batch = generator.generate_with_gillespie(
                            x=x_batch,
                            t=t_batch,
                            x_sizes=x_sizes_batch,
                            temperature=1.0,
                            no_special_toks=True,
                            max_decode_steps=1000,
                            use_scalar_steps=False,
                            verbose=False,
                        )

                # Convert batch to strings
                for i in range(current_batch_size):
                    y_str = "".join([vocab.token(idx.item()) for idx in y_batch[i]
                                    if vocab.token(idx.item()) in set("ARNDCQEGHILKMFPSTWYV")])
                    sampled_seqs.append(y_str)

            method_time = time.time() - start_time
            time_per_sample = method_time / num_samples
            print(f"  Time: {method_time:.1f}s ({time_per_sample:.2f}s per sample)")

            # Score samples
            print(f"Scoring {method} samples...")
            sampled_scores, _ = oracle.predict_batch(sampled_seqs)
            sampled_deltas = sampled_scores - seed_fitness
            sampled_mutations = [count_mutations(seq, seed_seq_str) for seq in sampled_seqs]

            # Store individual sample results
            for seq, score, delta, n_mut in zip(sampled_seqs, sampled_scores, sampled_deltas, sampled_mutations):
                all_results.append({
                    "seed_idx": seed_idx,
                    "seed_seq": seed_seq_str,
                    "seed_fitness": seed_fitness,
                    "method": method,
                    "sampled_seq": seq,
                    "fitness": score,
                    "fitness_delta": delta,
                    "num_mutations": n_mut,
                })

            # Store runtime summary
            runtime_data.append({
                "seed_idx": seed_idx,
                "method": method,
                "num_samples": num_samples,
                "total_time_seconds": method_time,
                "time_per_sample_seconds": time_per_sample,
            })

            # Print summary
            print(f"  Mean Δ: {np.mean(sampled_deltas):+.4f} ± {np.std(sampled_deltas):.4f}")
            print(f"  % Improved: {(sampled_deltas > 0).sum() / len(sampled_deltas) * 100:.1f}%")

    # Save all results to experiment-specific directory
    df = pd.DataFrame(all_results)
    runtime_df = pd.DataFrame(runtime_data)

    results_csv = exp_dir / "results.csv"
    runtime_csv = exp_dir / "runtime_data.csv"

    df.to_csv(results_csv, index=False)
    runtime_df.to_csv(runtime_csv, index=False)

    total_time = time.time() - total_start_time
    print(f"\n{'='*80}")
    print(f"Total runtime: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Results saved to: {results_csv}")
    print(f"Runtime data saved to: {runtime_csv}")
    print(f"{'='*80}")

    # Compute summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS BY METHOD")
    print(f"{'='*80}")

    summary_stats = []
    for method in methods:
        print(f"\n{method.upper()}:")
        method_df = df[df['method'] == method]

        if len(method_df) > 0:
            mean_delta = method_df['fitness_delta'].mean()
            std_delta = method_df['fitness_delta'].std()
            median_delta = method_df['fitness_delta'].median()
            pct_improved = (method_df['fitness_delta'] > 0).sum() / len(method_df) * 100
            mean_mutations = method_df['num_mutations'].mean()

            # Get average runtime
            runtime_info = runtime_df[runtime_df['method'] == method]
            avg_time_per_sample = runtime_info['time_per_sample_seconds'].mean()

            print(f"  Mean Δ: {mean_delta:+.4f} ± {std_delta:.4f}")
            print(f"  Median Δ: {median_delta:+.4f}")
            print(f"  % Improved: {pct_improved:.1f}%")
            print(f"  Mean mutations: {mean_mutations:.1f}")
            print(f"  Avg time/sample: {avg_time_per_sample:.2f}s")

            summary_stats.append({
                'method': method,
                'mean_delta': mean_delta,
                'std_delta': std_delta,
                'median_delta': median_delta,
                'pct_improved': pct_improved,
                'mean_mutations': mean_mutations,
                'avg_time_per_sample': avg_time_per_sample,
            })

    summary_df = pd.DataFrame(summary_stats)
    summary_csv = exp_dir / "summary_stats.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nSummary statistics saved to: {summary_csv}")

    # Statistical comparisons
    if len(methods) > 1:
        from scipy import stats
        print(f"\n{'='*80}")
        print("STATISTICAL COMPARISONS")
        print(f"{'='*80}")

        for i, method1 in enumerate(methods):
            for method2 in methods[i+1:]:
                df1 = df[df['method'] == method1]
                df2 = df[df['method'] == method2]

                if len(df1) > 0 and len(df2) > 0:
                    mean1 = df1['fitness_delta'].mean()
                    mean2 = df2['fitness_delta'].mean()
                    difference = mean1 - mean2
                    t_stat, p_value = stats.ttest_ind(df1['fitness_delta'], df2['fitness_delta'])

                    print(f"\n{method1.upper()} vs {method2.upper()}:")
                    print(f"  {method1} mean Δ: {mean1:+.4f}")
                    print(f"  {method2} mean Δ: {mean2:+.4f}")
                    print(f"  Difference: {difference:+.4f}")
                    print(f"  T-test: t={t_stat:.4f}, p={p_value:.4e}")
                    if p_value < 0.05:
                        print(f"  ✓ Significantly different (p < 0.05)")
                    else:
                        print(f"  ✗ Not significantly different (p >= 0.05)")

    # Save experiment metadata
    metadata = {
        'timestamp': timestamp,
        'variant': variant,
        'branch_length': branch_length,
        'methods': methods,
        'num_samples': num_samples,
        'num_seeds': len(seed_sequences),
        'total_available_seeds': total_available_seeds,
        'guidance_strength': guidance_strength,
        'random_seed': random_seed,
        'total_runtime_seconds': total_time,
        'experiment_name': exp_name,
    }
    metadata_df = pd.DataFrame([metadata])
    metadata_csv = exp_dir / "experiment_metadata.csv"
    metadata_df.to_csv(metadata_csv, index=False)
    print(f"Experiment metadata saved to: {metadata_csv}")

    # Create visualizations
    print(f"\n{'='*80}")
    print("GENERATING PLOTS")
    print(f"{'='*80}")

    create_icml_plots(df, methods, exp_dir)

    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"\nAll results saved to: {exp_dir}")


def create_icml_plots(df, methods, output_dir):
    """Create ICML-style publication-ready plots."""

    # ICML style - slightly narrower width
    figsize = (3.0, 2.5)

    # Color scheme
    method_colors = {
        'unguided': '#0173B2',  # Blue
        'exact_guided': '#E69F00',  # Orange/amber
        'taylor_guided': '#029E73',  # Green
    }

    positions = np.arange(len(methods))

    # ========================================================================
    # PLOT 1: Fitness Delta
    # ========================================================================
    plt.figure(figsize=figsize)

    bp = plt.boxplot(
        [df[df['method'] == m]['fitness_delta'].values for m in methods],
        positions=positions,
        widths=0.6,
        patch_artist=True,
        showfliers=False,
        boxprops=dict(linewidth=1.5),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
        medianprops=dict(linewidth=1.5, color='darkred'),
    )

    # Color boxes
    for patch, method in zip(bp['boxes'], methods):
        patch.set_facecolor(method_colors.get(method, '#888888'))
        patch.set_alpha(0.7)

    plt.axhline(0, color='black', linestyle='--', linewidth=1.0, alpha=0.5)
    plt.xlabel('Method', fontsize=10)
    plt.ylabel('Δ Fitness', fontsize=10)
    plt.grid(True, alpha=0.2, linewidth=0.5)
    plt.xticks(positions, [m.replace('_', ' ').title() for m in methods], fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout(pad=0.3)

    # Save
    plot_path = output_dir / 'fitness_delta.pdf'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    png_path = output_dir / 'fitness_delta.png'
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    print(f"Saved {plot_path} and {png_path}")
    plt.close()

    # ========================================================================
    # PLOT 2: Number of Mutations
    # ========================================================================
    plt.figure(figsize=figsize)

    bp = plt.boxplot(
        [df[df['method'] == m]['num_mutations'].values for m in methods],
        positions=positions,
        widths=0.6,
        patch_artist=True,
        showfliers=False,
        boxprops=dict(linewidth=1.5),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
        medianprops=dict(linewidth=1.5, color='darkred'),
    )

    # Color boxes
    for patch, method in zip(bp['boxes'], methods):
        patch.set_facecolor(method_colors.get(method, '#888888'))
        patch.set_alpha(0.7)

    plt.xlabel('Method', fontsize=10)
    plt.ylabel('Number of Mutations', fontsize=10)
    plt.grid(True, alpha=0.2, linewidth=0.5)
    plt.xticks(positions, [m.replace('_', ' ').title() for m in methods], fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout(pad=0.3)

    # Save
    plot_path = output_dir / 'num_mutations.pdf'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    png_path = output_dir / 'num_mutations.png'
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    print(f"Saved {plot_path} and {png_path}")
    plt.close()

    # Print summary
    print("\nPlot Summary:")
    for method in methods:
        method_df = df[df['method'] == method]
        print(f"  {method.upper()}: Δ Fitness = {method_df['fitness_delta'].mean():+.4f} ± {method_df['fitness_delta'].std():.4f}")


if __name__ == "__main__":
    main()
