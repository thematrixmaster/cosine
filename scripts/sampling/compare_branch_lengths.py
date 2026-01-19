"""Compare Taylor vs Exact guidance across different branch lengths.

This script sweeps over multiple branch lengths to characterize how guidance
performance changes with evolutionary distance.
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
import seaborn as sns
from tqdm import tqdm
import time
import random
from datetime import datetime

from evo.oracles import get_oracle, CovidOracle
from evo.tokenization import Vocab
from evo.antibody import compute_iglm_humanness, compute_oasis_humanness
from peint.models.modules.ctmc_module import CTMCModule
from peint.models.nets.ctmc import NeuralCTMC, NeuralCTMCGenerator


def compute_humanness_score(seqs, *args, **kwargs):
    # return compute_iglm_humanness(seqs, *args, **kwargs)
    return compute_oasis_humanness(seqs, *args, **kwargs)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compare Taylor vs Exact guidance across branch lengths'
    )
    parser.add_argument('--branch-lengths', nargs='+', type=float,
                        default=[0.1, 0.5, 1.0],
                        help='Branch lengths to test (default: 0.1 0.5 1.0)')
    parser.add_argument('--num-samples', type=int, default=10,
                        help='Number of samples per (branch_length, method) combination (default: 10)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size for sampling (default: same as num-samples)')
    parser.add_argument('--methods', nargs='+',
                        choices=['unguided', 'exact_guided', 'taylor_guided'],
                        default=['exact_guided', 'taylor_guided'],
                        help='Methods to compare (default: exact_guided taylor_guided)')
    parser.add_argument('--guidance-strength', type=float, default=2.0,
                        help='Guidance strength γ for guided methods (default: 2.0)')
    parser.add_argument('--output-dir', type=str,
                        default='/scratch/users/stephen.lu/projects/protevo/results/branch_length_comparison',
                        help='Output directory for results')
    parser.add_argument('--model-path', type=str,
                        default='/scratch/users/stephen.lu/projects/protevo/logs/train/runs/2026-01-06_18-32-49/checkpoints/epoch_001.ckpt',
                        help='Path to CTMC model checkpoint')
    parser.add_argument('--oracle-chunk-size', type=int, default=5000,
                        help='Max sequences per oracle call to avoid OOM (default: 5000)')
    parser.add_argument('--num-seeds', type=int, default=None,
                        help='Number of seed sequences to use (default: all available seeds)')
    parser.add_argument('--num-mc-samples', type=int, default=10,
                        help='Number of MC samples for oracle (default: 10)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility (default: None, uses timestamp)')
    parser.add_argument('--variant', type=str, default='SARSCoV1',
                        choices=['SARSCoV1', 'SARSCoV2Beta'],
                        help='Oracle variant to use (default: SARSCoV1)')
    parser.add_argument('--plot-only', action='store_true',
                        help='Only regenerate plots from existing results (no sampling)')
    parser.add_argument('--results-dir', type=str, default=None,
                        help='Directory containing results.csv for --plot-only mode')
    return parser.parse_args()


def main():
    args = parse_args()

    # Plot-only mode: regenerate plots from saved results
    if args.plot_only:
        if args.results_dir is None:
            print("Error: --results-dir required for --plot-only mode")
            sys.exit(1)

        results_dir = Path(args.results_dir)
        results_csv = results_dir / "results.csv"

        if not results_csv.exists():
            print(f"Error: Results file not found: {results_csv}")
            sys.exit(1)

        print(f"Loading results from: {results_csv}")
        df = pd.DataFrame(pd.read_csv(results_csv))

        # Extract branch lengths and methods from data
        branch_lengths = sorted(df['branch_length'].unique())
        methods = sorted(df['method'].unique())

        print(f"\nRegenerating plots for {len(branch_lengths)} branch lengths and {len(methods)} methods...")
        create_performance_plot(df, methods, branch_lengths, results_dir)
        create_runtime_plot_from_results(df, methods, branch_lengths, results_dir)

        print(f"\nPlots regenerated in: {results_dir}")
        return

    # Normal sampling mode
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
    branch_lengths = sorted(args.branch_lengths)
    methods = args.methods
    num_samples = args.num_samples
    batch_size = args.batch_size if args.batch_size is not None else num_samples
    guidance_strength = args.guidance_strength
    oracle_chunk_size = args.oracle_chunk_size
    variant = args.variant
    num_mc_samples = args.num_mc_samples
    num_seeds = args.num_seeds
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create experiment-specific subdirectory to avoid overwriting
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    branch_str = "_".join([f"{bl:.2f}".replace(".", "p") for bl in branch_lengths])
    method_str = "_".join([m.replace("_guided", "") for m in methods])
    exp_name = f"{timestamp}_{variant}_bl{branch_str}_{method_str}_n{num_samples}_seed{random_seed}"
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

    # Load oracle on CUDA with MC Dropout for uncertainty
    print(f"Loading oracle ({variant}) on {device} with MC Dropout...")
    oracle: CovidOracle = get_oracle(
        variant,
        enable_mc_dropout=True,
        mc_samples=num_mc_samples,
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
        # Random sampling with the set seed for reproducibility
        seed_indices = random.sample(range(total_available_seeds), num_seeds)
        seed_sequences = [all_seed_sequences[i] for i in sorted(seed_indices)]
        print(f"Selected {num_seeds} seeds from {total_available_seeds} available (indices: {sorted(seed_indices)[:10]}{'...' if len(seed_indices) > 10 else ''})")
    else:
        seed_sequences = all_seed_sequences
        print(f"Using all {total_available_seeds} available seed sequences")

    print(f"\n{'='*80}")
    print("BRANCH LENGTH COMPARISON: TAYLOR VS EXACT GUIDANCE")
    print(f"{'='*80}")
    print(f"Variant: {variant}")
    print(f"Branch lengths: {branch_lengths}")
    print(f"Methods: {', '.join(methods)}")
    print(f"Samples per (branch_length, method): {num_samples}")
    print(f"Guidance strength (γ): {guidance_strength}")
    print(f"Random seed: {random_seed}")
    print(f"Number of seed sequences: {len(seed_sequences)}")
    print(f"Oracle: {variant} neutralization with MC Dropout ({num_mc_samples} samples)")
    print(f"\nTotal trajectories: {len(branch_lengths) * len(methods) * num_samples * len(seed_sequences)}")
    print(f"  {len(branch_lengths)} branch lengths × {len(methods)} methods × {num_samples} samples × {len(seed_sequences)} seeds")
    print(f"{'='*80}\n")

    # Compute seed humanness scores once (this can be slow)
    print("Computing humanness for all seed sequences...")
    seed_seqs_list = [s["sequence"] for s in seed_sequences]
    seed_humanness_scores = compute_humanness_score(seed_seqs_list, chain="heavy")

    # Update seed_sequences with humanness scores
    for i, humanness in enumerate(seed_humanness_scores):
        seed_sequences[i]["humanness"] = humanness

    print(f"Seed humanness scores computed (mean: {np.mean(seed_humanness_scores):.4f})\n")

    # Estimate time
    samples_per_method = num_samples * len(seed_sequences)
    if 'exact_guided' in methods and 'taylor_guided' in methods:
        est_time = len(branch_lengths) * (samples_per_method * 60 + samples_per_method * 1)
    elif 'exact_guided' in methods:
        est_time = len(branch_lengths) * samples_per_method * 60
    else:
        est_time = len(branch_lengths) * samples_per_method * 1.5
    print(f"\nEstimated time: ~{est_time/60:.0f} minutes")
    print(f"{'='*80}\n")

    all_results = []
    runtime_data = []
    total_start_time = time.time()

    # Helper function to count mutations
    def count_mutations(seq, seed):
        min_len = min(len(seq), len(seed))
        return sum(1 for i in range(min_len) if seq[i] != seed[i])

    # Outer loop over branch lengths
    for branch_length in branch_lengths:
        print(f"\n{'#'*80}")
        print(f"BRANCH LENGTH: {branch_length}")
        print(f"{'#'*80}")

        # Loop over seeds
        for seed_idx, seed_data in enumerate(seed_sequences):
            seed_seq_str = seed_data["sequence"]
            seed_fitness = seed_data["fitness"]
            seed_humanness = seed_data["humanness"]

            print(f"\n{'='*80}")
            print(f"Branch Length: {branch_length} | Seed {seed_idx}: {seed_seq_str[:40]}...")
            print(f"Seed fitness: {seed_fitness:.4f} | Seed humanness: {seed_humanness:.4f}")
            print(f"{'='*80}")

            # Convert seed to tensor
            x = torch.tensor([vocab.tokens_to_idx[aa] for aa in seed_seq_str], device=device).unsqueeze(0)
            x_sizes = torch.tensor([len(seed_seq_str)], device=device)
            t = torch.tensor([branch_length], device=device)

            # Sample for each method
            for method in methods:
                print(f"\n[{method.upper()}] Sampling {num_samples} sequences (batch_size={batch_size})...")
                start_time = time.time()

                sampled_seqs = []

                # Process in batches
                for batch_start in range(0, num_samples, batch_size):
                    batch_end = min(batch_start + batch_size, num_samples)
                    current_batch_size = batch_end - batch_start

                    # Replicate seed sequence for this batch
                    x_batch = x.repeat(current_batch_size, 1)
                    t_batch = t.repeat(current_batch_size)
                    x_sizes_batch = x_sizes.repeat(current_batch_size)

                    with (torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16)):
                        if method == 'unguided':
                            y_batch = generator.generate_with_adapted_gillespie(
                                x=x_batch,
                                t=t_batch,
                                x_sizes=x_sizes_batch,
                                temperature=1.0,
                                no_special_toks=True,
                                max_decode_steps=1000,
                                use_scalar_steps=False,
                                verbose=False,
                            )
                        elif method in ['exact_guided', 'taylor_guided']:
                            use_taylor = (method == 'taylor_guided')
                            y_batch = generator.generate_with_guided_gillespie(
                                x=x_batch,
                                t=t_batch,
                                x_sizes=x_sizes_batch,
                                oracle=oracle,
                                guidance_strength=guidance_strength,
                                temperature=1.0,
                                no_special_toks=True,
                                max_decode_steps=1000,
                                use_scalar_steps=False,
                                use_taylor_approx=use_taylor,
                                verbose=(batch_start == 0),
                                oracle_chunk_size=oracle_chunk_size,
                            )

                    # Convert batch to strings
                    for i in range(current_batch_size):
                        y_str = "".join([vocab.token(idx.item()) for idx in y_batch[i]
                                        if vocab.token(idx.item()) in set("ARNDCQEGHILKMFPSTWYV")])
                        sampled_seqs.append(y_str)

                method_time = time.time() - start_time
                time_per_sample = method_time / num_samples
                print(f"  Sampling time: {method_time:.1f}s ({time_per_sample:.2f}s per sample)")

                # Score samples with oracle
                print(f"Scoring {method} samples with oracle...")
                sampled_scores, _ = oracle.predict_batch(sampled_seqs)
                sampled_fitness_deltas = sampled_scores - seed_fitness

                # Compute humanness for samples
                print(f"Computing humanness for samples...")
                humanness_start = time.time()
                sampled_humanness = compute_humanness_score(sampled_seqs, chain="heavy")
                sampled_humanness_deltas = np.array(sampled_humanness) - seed_humanness
                humanness_time = time.time() - humanness_start
                print(f"  Humanness computation: {humanness_time:.1f}s")

                # Count mutations
                sampled_mutations = [count_mutations(seq, seed_seq_str) for seq in sampled_seqs]

                # Store individual sample results
                for seq, fit, fit_delta, human, human_delta, n_mut in zip(
                    sampled_seqs, sampled_scores, sampled_fitness_deltas,
                    sampled_humanness, sampled_humanness_deltas, sampled_mutations
                ):
                    all_results.append({
                        "branch_length": branch_length,
                        "seed_idx": seed_idx,
                        "seed_seq": seed_seq_str,
                        "seed_fitness": seed_fitness,
                        "seed_humanness": seed_humanness,
                        "method": method,
                        "sampled_seq": seq,
                        "fitness": fit,
                        "fitness_delta": fit_delta,
                        "humanness": human,
                        "humanness_delta": human_delta,
                        "num_mutations": n_mut,
                        "guidance_strength": guidance_strength,
                    })

                # Store runtime summary for this combination
                runtime_data.append({
                    "branch_length": branch_length,
                    "seed_idx": seed_idx,
                    "method": method,
                    "num_samples": num_samples,
                    "sampling_time_seconds": method_time,
                    "humanness_time_seconds": humanness_time,
                    "total_time_seconds": method_time + humanness_time,
                    "time_per_sample_seconds": (method_time + humanness_time) / num_samples,
                })

                # Print summary for this combination
                print(f"  Mean Δ fitness: {np.mean(sampled_fitness_deltas):+.4f} ± {np.std(sampled_fitness_deltas):.4f}")
                print(f"  Mean Δ humanness: {np.mean(sampled_humanness_deltas):+.4f} ± {np.std(sampled_humanness_deltas):.4f}")
                print(f"  % Fitness improved: {(sampled_fitness_deltas > 0).sum() / len(sampled_fitness_deltas) * 100:.1f}%")

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
    print("SUMMARY STATISTICS BY BRANCH LENGTH AND METHOD")
    print(f"{'='*80}")

    summary_stats = []
    for branch_length in branch_lengths:
        print(f"\nBranch Length: {branch_length}")
        print("-" * 60)
        for method in methods:
            method_df = df[(df['branch_length'] == branch_length) & (df['method'] == method)]
            if len(method_df) > 0:
                mean_fitness_delta = method_df['fitness_delta'].mean()
                std_fitness_delta = method_df['fitness_delta'].std()
                mean_humanness_delta = method_df['humanness_delta'].mean()
                std_humanness_delta = method_df['humanness_delta'].std()
                median_fitness_delta = method_df['fitness_delta'].median()
                pct_fitness_improved = (method_df['fitness_delta'] > 0).sum() / len(method_df) * 100
                mean_mutations = method_df['num_mutations'].mean()

                # Get average runtime
                runtime_info = runtime_df[(runtime_df['branch_length'] == branch_length) &
                                         (runtime_df['method'] == method)]
                avg_time_per_sample = runtime_info['time_per_sample_seconds'].mean()

                print(f"  {method.upper()}:")
                print(f"    Mean Δ fitness: {mean_fitness_delta:+.4f} ± {std_fitness_delta:.4f}")
                print(f"    Mean Δ humanness: {mean_humanness_delta:+.4f} ± {std_humanness_delta:.4f}")
                print(f"    Median Δ fitness: {median_fitness_delta:+.4f}")
                print(f"    % Fitness improved: {pct_fitness_improved:.1f}%")
                print(f"    Mean mutations: {mean_mutations:.1f}")
                print(f"    Avg time/sample: {avg_time_per_sample:.2f}s")

                summary_stats.append({
                    'branch_length': branch_length,
                    'method': method,
                    'mean_fitness_delta': mean_fitness_delta,
                    'std_fitness_delta': std_fitness_delta,
                    'mean_humanness_delta': mean_humanness_delta,
                    'std_humanness_delta': std_humanness_delta,
                    'median_fitness_delta': median_fitness_delta,
                    'pct_fitness_improved': pct_fitness_improved,
                    'mean_mutations': mean_mutations,
                    'avg_time_per_sample': avg_time_per_sample,
                })

    summary_df = pd.DataFrame(summary_stats)
    summary_csv = exp_dir / "summary_stats.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nSummary statistics saved to: {summary_csv}")

    # Statistical comparisons
    if 'exact_guided' in methods and 'taylor_guided' in methods:
        from scipy import stats
        print(f"\n{'='*80}")
        print("STATISTICAL COMPARISON: TAYLOR VS EXACT")
        print(f"{'='*80}")

        for branch_length in branch_lengths:
            exact_df = df[(df['branch_length'] == branch_length) & (df['method'] == 'exact_guided')]
            taylor_df = df[(df['branch_length'] == branch_length) & (df['method'] == 'taylor_guided')]

            if len(exact_df) > 0 and len(taylor_df) > 0:
                # Fitness comparison
                t_stat_fit, p_value_fit = stats.ttest_ind(exact_df['fitness_delta'], taylor_df['fitness_delta'])
                exact_mean_fit = exact_df['fitness_delta'].mean()
                taylor_mean_fit = taylor_df['fitness_delta'].mean()
                difference_fit = exact_mean_fit - taylor_mean_fit

                # Humanness comparison
                t_stat_human, p_value_human = stats.ttest_ind(exact_df['humanness_delta'], taylor_df['humanness_delta'])
                exact_mean_human = exact_df['humanness_delta'].mean()
                taylor_mean_human = taylor_df['humanness_delta'].mean()
                difference_human = exact_mean_human - taylor_mean_human

                print(f"\nBranch Length: {branch_length}")
                print(f"  FITNESS:")
                print(f"    Exact mean Δ: {exact_mean_fit:+.4f}")
                print(f"    Taylor mean Δ: {taylor_mean_fit:+.4f}")
                print(f"    Difference (Exact - Taylor): {difference_fit:+.4f}")
                print(f"    T-test: t={t_stat_fit:.4f}, p={p_value_fit:.4e}")
                if p_value_fit < 0.05:
                    print(f"    ✓ Significantly different (p < 0.05)")
                else:
                    print(f"    ✗ Not significantly different (p >= 0.05)")

                print(f"  HUMANNESS:")
                print(f"    Exact mean Δ: {exact_mean_human:+.4f}")
                print(f"    Taylor mean Δ: {taylor_mean_human:+.4f}")
                print(f"    Difference (Exact - Taylor): {difference_human:+.4f}")
                print(f"    T-test: t={t_stat_human:.4f}, p={p_value_human:.4e}")
                if p_value_human < 0.05:
                    print(f"    ✓ Significantly different (p < 0.05)")
                else:
                    print(f"    ✗ Not significantly different (p >= 0.05)")

    # Save experiment metadata
    metadata = {
        'timestamp': timestamp,
        'variant': variant,
        'branch_lengths': branch_lengths,
        'methods': methods,
        'num_samples': num_samples,
        'num_seeds': len(seed_sequences),
        'total_available_seeds': total_available_seeds,
        'num_mc_samples': num_mc_samples,
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

    # Plot 1: Performance comparison (boxplot)
    create_performance_plot(df, methods, branch_lengths, exp_dir)

    # Plot 2: Runtime comparison
    create_runtime_plot(runtime_df, methods, branch_lengths, exp_dir)

    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"\nAll results saved to: {exp_dir}")
    print(f"\nTo regenerate plots with different settings:")
    print(f"  uv run python {Path(__file__).name} --plot-only --results-dir {exp_dir}")


def create_performance_plot(df, methods, branch_lengths, output_dir):
    """Create side-by-side boxplot comparing performance across branch lengths."""

    # Set up plot style
    sns.set_style("whitegrid")
    plt.rcParams['font.size'] = 11

    fig, ax = plt.subplots(figsize=(12, 7))

    # Prepare data for plotting
    plot_data = []
    for branch_length in branch_lengths:
        for method in methods:
            method_df = df[(df['branch_length'] == branch_length) & (df['method'] == method)]
            if len(method_df) > 0:
                for delta in method_df['fitness_delta']:
                    plot_data.append({
                        'Branch Length': branch_length,
                        'Method': method.replace('_', ' ').title(),
                        'Δ Fitness': delta
                    })

    plot_df = pd.DataFrame(plot_data)

    # Define colors
    if 'exact_guided' in methods and 'taylor_guided' in methods:
        palette = {'Exact Guided': '#e74c3c', 'Taylor Guided': '#27ae60'}
    elif 'exact_guided' in methods:
        palette = {'Exact Guided': '#e74c3c'}
    elif 'taylor_guided' in methods:
        palette = {'Taylor Guided': '#27ae60'}
    else:
        palette = None

    # Create boxplot
    sns.boxplot(
        data=plot_df,
        x='Branch Length',
        y='Δ Fitness',
        hue='Method',
        palette=palette,
        ax=ax,
        showfliers=False,  # Don't show outliers for cleaner look
    )

    # Add individual points with jitter
    for i, branch_length in enumerate(branch_lengths):
        for j, method in enumerate(methods):
            method_name = method.replace('_', ' ').title()
            method_df = plot_df[(plot_df['Branch Length'] == branch_length) &
                               (plot_df['Method'] == method_name)]
            if len(method_df) > 0:
                x_pos = i + (j - 0.5) * 0.25  # Adjust x position for side-by-side
                y_vals = method_df['Δ Fitness'].values
                x_vals = np.random.normal(x_pos, 0.05, size=len(y_vals))  # Add jitter

                color = palette[method_name] if palette else 'gray'
                ax.scatter(x_vals, y_vals, alpha=0.3, s=20, color=color, zorder=10)

    # Formatting
    ax.axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label='No change')
    ax.set_xlabel('Branch Length (evolutionary distance)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Δ Oracle Score (fitness change)', fontsize=13, fontweight='bold')
    ax.set_title('Performance Comparison: Taylor vs Exact Guidance\nAcross Branch Lengths',
                 fontsize=15, fontweight='bold', pad=20)

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels, loc='best', frameon=True,
              shadow=True, fontsize=11, title='Method', title_fontsize=12)

    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    # Save
    performance_plot = output_dir / "performance_comparison.png"
    plt.savefig(performance_plot, dpi=300, bbox_inches='tight')
    print(f"Performance plot saved to: {performance_plot}")
    plt.close()


def create_runtime_plot_from_results(df, methods, branch_lengths, output_dir):
    """Create runtime plot from results (for plot-only mode)."""
    # This is a placeholder - in plot-only mode we don't have runtime data
    # So we skip runtime plotting
    print("Skipping runtime plot (no runtime data in results.csv for plot-only mode)")
    return


def create_runtime_plot(runtime_df, methods, branch_lengths, output_dir):
    """Create bar chart comparing runtime across methods and branch lengths."""

    # Set up plot style
    sns.set_style("whitegrid")
    plt.rcParams['font.size'] = 11

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Aggregate runtime data (average across seeds)
    agg_runtime = runtime_df.groupby(['branch_length', 'method'])['time_per_sample_seconds'].mean().reset_index()

    # Define colors
    if 'exact_guided' in methods and 'taylor_guided' in methods:
        palette = {'exact_guided': '#e74c3c', 'taylor_guided': '#27ae60'}
    elif 'exact_guided' in methods:
        palette = {'exact_guided': '#e74c3c'}
    elif 'taylor_guided' in methods:
        palette = {'taylor_guided': '#27ae60'}
    else:
        palette = None

    # Plot 1: Time per sample (bar chart)
    x = np.arange(len(branch_lengths))
    width = 0.35

    for i, method in enumerate(methods):
        method_data = agg_runtime[agg_runtime['method'] == method]
        times = [method_data[method_data['branch_length'] == bl]['time_per_sample_seconds'].values[0]
                 if len(method_data[method_data['branch_length'] == bl]) > 0 else 0
                 for bl in branch_lengths]

        offset = (i - len(methods)/2 + 0.5) * width
        bars = ax1.bar(x + offset, times, width,
                      label=method.replace('_', ' ').title(),
                      color=palette[method] if palette else None,
                      alpha=0.8)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}s',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax1.set_xlabel('Branch Length', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Time per Sample (seconds)', fontsize=13, fontweight='bold')
    ax1.set_title('Runtime Comparison: Time per Sample', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(branch_lengths)
    ax1.legend(loc='best', frameon=True, shadow=True, fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')

    # Plot 2: Speedup factor (if both methods present)
    if 'exact_guided' in methods and 'taylor_guided' in methods:
        speedup_data = []
        for bl in branch_lengths:
            exact_time = agg_runtime[(agg_runtime['branch_length'] == bl) &
                                     (agg_runtime['method'] == 'exact_guided')]['time_per_sample_seconds'].values
            taylor_time = agg_runtime[(agg_runtime['branch_length'] == bl) &
                                      (agg_runtime['method'] == 'taylor_guided')]['time_per_sample_seconds'].values
            if len(exact_time) > 0 and len(taylor_time) > 0:
                speedup = exact_time[0] / taylor_time[0]
                speedup_data.append(speedup)
            else:
                speedup_data.append(0)

        bars = ax2.bar(x, speedup_data, width=0.6, color='#3498db', alpha=0.8)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}×',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax2.axhline(1, color='black', linestyle='--', linewidth=1.5, alpha=0.7,
                   label='No speedup')
        ax2.set_xlabel('Branch Length', fontsize=13, fontweight='bold')
        ax2.set_ylabel('Speedup Factor (Exact / Taylor)', fontsize=13, fontweight='bold')
        ax2.set_title('Taylor Approximation Speedup', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(branch_lengths)
        ax2.legend(loc='best', frameon=True, shadow=True, fontsize=11)
        ax2.grid(True, alpha=0.3, axis='y')
    else:
        # If only one method, show total time instead
        for i, method in enumerate(methods):
            method_data = agg_runtime[agg_runtime['method'] == method]
            # Calculate total time for typical experiment (50 samples × 2 seeds)
            total_times = [method_data[method_data['branch_length'] == bl]['time_per_sample_seconds'].values[0] * 50 * 2
                          if len(method_data[method_data['branch_length'] == bl]) > 0 else 0
                          for bl in branch_lengths]

            bars = ax2.bar(x, total_times, width=0.6,
                          label=method.replace('_', ' ').title(),
                          color=palette[method] if palette else None,
                          alpha=0.8)

            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height/60:.1f}m',
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax2.set_xlabel('Branch Length', fontsize=13, fontweight='bold')
        ax2.set_ylabel('Total Time (seconds)', fontsize=13, fontweight='bold')
        ax2.set_title('Estimated Total Runtime (100 samples)', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(branch_lengths)
        ax2.legend(loc='best', frameon=True, shadow=True, fontsize=11)
        ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # Save
    runtime_plot = output_dir / "runtime_comparison.png"
    plt.savefig(runtime_plot, dpi=300, bbox_inches='tight')
    print(f"Runtime plot saved to: {runtime_plot}")
    plt.close()


if __name__ == "__main__":
    main()
