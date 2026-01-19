"""Sweep TAG guidance strengths to characterize fitness-humanness tradeoff.

This script tests Taylor Series Approximation Guidance (TAG) with different
guidance strengths to measure the tradeoff between fitness improvement and
humanness (IGLM log-likelihood).
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
        description='Sweep TAG guidance strengths to measure fitness-humanness tradeoff'
    )
    parser.add_argument('--guidance-strengths', nargs='+', type=float,
                        default=[0.5, 1.0, 2.0, 4.0],
                        help='Guidance strengths to test (default: 0.5 1.0 2.0 4.0)')
    parser.add_argument('--branch-length', type=float, default=0.5,
                        help='Branch length for sampling (default: 0.5)')
    parser.add_argument('--num-samples', type=int, default=10,
                        help='Number of samples per guidance strength (default: 10)')
    parser.add_argument('--num-mc-samples', type=int, default=10,
                        help='Number of MC samples for oracle (default: 10)')
    parser.add_argument('--batch-size', type=int, default=25,
                        help='Batch size for sampling (default: 25)')
    parser.add_argument('--output-dir', type=str,
                        default='/scratch/users/stephen.lu/projects/protevo/results/tag_guidance_sweep',
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

        # Extract guidance strengths from data
        guidance_strengths = sorted(df['guidance_strength'].unique())

        print(f"\nRegenerating plots for {len(guidance_strengths)} guidance strengths...")
        create_scatter_plot(df, guidance_strengths, results_dir)
        create_summary_plots(df, guidance_strengths, results_dir)

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
    guidance_strengths = sorted(args.guidance_strengths)
    branch_length = args.branch_length
    num_samples = args.num_samples
    batch_size = args.batch_size if args.batch_size is not None else num_samples
    oracle_chunk_size = args.oracle_chunk_size
    variant = args.variant
    num_mc_samples = args.num_mc_samples
    num_seeds = args.num_seeds
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create experiment-specific subdirectory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    gamma_str = "_".join([f"{g:.2f}".replace(".", "p") for g in guidance_strengths])
    bl_str = f"{branch_length:.2f}".replace(".", "p")
    exp_name = f"{timestamp}_TAG_gamma{gamma_str}_bl{bl_str}_n{num_samples}_seed{random_seed}"
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
        seed_indices = random.sample(range(total_available_seeds), num_seeds)
        seed_sequences = [all_seed_sequences[i] for i in sorted(seed_indices)]
        print(f"Selected {num_seeds} seeds from {total_available_seeds} available")
    else:
        seed_sequences = all_seed_sequences
        print(f"Using all {total_available_seeds} available seed sequences")

    print(f"\n{'='*80}")
    print("TAG GUIDANCE STRENGTH SWEEP: FITNESS-HUMANNESS TRADEOFF")
    print(f"{'='*80}")
    print(f"Variant: {variant}")
    print(f"Branch length: {branch_length}")
    print(f"Guidance strengths (γ): {guidance_strengths}")
    print(f"Samples per guidance strength: {num_samples}")
    print(f"Random seed: {random_seed}")
    print(f"Number of seed sequences: {len(seed_sequences)}")
    print(f"Oracle: {variant} neutralization with MC Dropout ({num_mc_samples} samples)")
    print(f"\nTotal trajectories: {len(guidance_strengths) * num_samples * len(seed_sequences)}")
    print(f"  {len(guidance_strengths)} gamma values × {num_samples} samples × {len(seed_sequences)} seeds")
    print(f"{'='*80}\n")

    # Compute seed humanness scores once (this can be slow)
    print("Computing humanness for all seed sequences...")
    seed_seqs_list = [s["sequence"] for s in seed_sequences]
    seed_humanness_scores = compute_humanness_score(seed_seqs_list, chain="heavy")

    # Update seed_sequences with humanness scores
    for i, humanness in enumerate(seed_humanness_scores):
        seed_sequences[i]["humanness"] = humanness

    print(f"Seed humanness scores computed (mean: {np.mean(seed_humanness_scores):.4f})\n")

    all_results = []
    runtime_data = []
    total_start_time = time.time()

    # Helper function to count mutations
    def count_mutations(seq, seed):
        min_len = min(len(seq), len(seed))
        return sum(1 for i in range(min_len) if seq[i] != seed[i])

    # Outer loop over guidance strengths
    for gamma in guidance_strengths:
        print(f"\n{'#'*80}")
        print(f"GUIDANCE STRENGTH (γ): {gamma}")
        print(f"{'#'*80}")

        # Loop over seeds
        for seed_idx, seed_data in enumerate(seed_sequences):
            seed_seq_str = seed_data["sequence"]
            seed_fitness = seed_data["fitness"]
            seed_humanness = seed_data["humanness"]

            print(f"\n{'='*80}")
            print(f"γ = {gamma} | Seed {seed_idx}: {seed_seq_str[:40]}...")
            print(f"Seed fitness: {seed_fitness:.4f} | Seed humanness: {seed_humanness:.4f}")
            print(f"{'='*80}")

            # Convert seed to tensor
            x = torch.tensor([vocab.tokens_to_idx[aa] for aa in seed_seq_str], device=device).unsqueeze(0)
            x_sizes = torch.tensor([len(seed_seq_str)], device=device)
            t = torch.tensor([branch_length], device=device)

            print(f"\n[TAG γ={gamma}] Sampling {num_samples} sequences (batch_size={batch_size})...")
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
                    y_batch = generator.generate_with_guided_gillespie(
                        x=x_batch,
                        t=t_batch,
                        x_sizes=x_sizes_batch,
                        oracle=oracle,
                        guidance_strength=gamma,
                        temperature=1.0,
                        no_special_toks=True,
                        max_decode_steps=1000,
                        use_scalar_steps=False,
                        use_taylor_approx=True,  # Use TAG
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
            print(f"Scoring samples with oracle...")
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
                    "guidance_strength": gamma,
                    "seed_idx": seed_idx,
                    "seed_seq": seed_seq_str,
                    "seed_fitness": seed_fitness,
                    "seed_humanness": seed_humanness,
                    "sampled_seq": seq,
                    "fitness": fit,
                    "fitness_delta": fit_delta,
                    "humanness": human,
                    "humanness_delta": human_delta,
                    "num_mutations": n_mut,
                })

            # Store runtime summary
            runtime_data.append({
                "guidance_strength": gamma,
                "seed_idx": seed_idx,
                "num_samples": num_samples,
                "sampling_time_seconds": method_time,
                "humanness_time_seconds": humanness_time,
                "total_time_seconds": method_time + humanness_time,
                "time_per_sample_seconds": (method_time + humanness_time) / num_samples,
            })

            # Print summary
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
    print("SUMMARY STATISTICS BY GUIDANCE STRENGTH")
    print(f"{'='*80}")

    summary_stats = []
    for gamma in guidance_strengths:
        print(f"\nGuidance Strength (γ): {gamma}")
        print("-" * 60)
        gamma_df = df[df['guidance_strength'] == gamma]

        if len(gamma_df) > 0:
            mean_fitness_delta = gamma_df['fitness_delta'].mean()
            std_fitness_delta = gamma_df['fitness_delta'].std()
            mean_humanness_delta = gamma_df['humanness_delta'].mean()
            std_humanness_delta = gamma_df['humanness_delta'].std()
            pct_fitness_improved = (gamma_df['fitness_delta'] > 0).sum() / len(gamma_df) * 100
            mean_mutations = gamma_df['num_mutations'].mean()

            # Get average runtime
            runtime_info = runtime_df[runtime_df['guidance_strength'] == gamma]
            avg_time_per_sample = runtime_info['time_per_sample_seconds'].mean()

            print(f"  Mean Δ fitness: {mean_fitness_delta:+.4f} ± {std_fitness_delta:.4f}")
            print(f"  Mean Δ humanness: {mean_humanness_delta:+.4f} ± {std_humanness_delta:.4f}")
            print(f"  % Fitness improved: {pct_fitness_improved:.1f}%")
            print(f"  Mean mutations: {mean_mutations:.1f}")
            print(f"  Avg time/sample: {avg_time_per_sample:.2f}s")

            summary_stats.append({
                'guidance_strength': gamma,
                'mean_fitness_delta': mean_fitness_delta,
                'std_fitness_delta': std_fitness_delta,
                'mean_humanness_delta': mean_humanness_delta,
                'std_humanness_delta': std_humanness_delta,
                'pct_fitness_improved': pct_fitness_improved,
                'mean_mutations': mean_mutations,
                'avg_time_per_sample': avg_time_per_sample,
            })

    summary_df = pd.DataFrame(summary_stats)
    summary_csv = exp_dir / "summary_stats.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nSummary statistics saved to: {summary_csv}")

    # Save experiment metadata
    metadata = {
        'timestamp': timestamp,
        'variant': variant,
        'branch_length': branch_length,
        'guidance_strengths': guidance_strengths,
        'num_samples': num_samples,
        'num_seeds': len(seed_sequences),
        'total_available_seeds': total_available_seeds,
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

    create_scatter_plot(df, guidance_strengths, exp_dir)
    create_summary_plots(df, guidance_strengths, exp_dir)

    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"\nAll results saved to: {exp_dir}")
    print(f"\nTo regenerate plots with different settings:")
    print(f"  uv run python {Path(__file__).name} --plot-only --results-dir {exp_dir}")


def create_scatter_plot(df, guidance_strengths, output_dir):
    """Create scatter plot: fitness_delta vs humanness_delta, colored by gamma."""

    sns.set_style("whitegrid")
    plt.rcParams['font.size'] = 11

    fig, ax = plt.subplots(figsize=(12, 9))

    # Color map for guidance strengths
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=min(guidance_strengths), vmax=max(guidance_strengths))

    # Plot each guidance strength with different color
    for gamma in guidance_strengths:
        subset = df[df['guidance_strength'] == gamma]

        if len(subset) > 0:
            scatter = ax.scatter(
                subset['humanness_delta'],
                subset['fitness_delta'],
                c=[gamma] * len(subset),
                cmap=cmap,
                norm=norm,
                alpha=0.6,
                s=80,
                label=f'γ = {gamma}',
                edgecolors='black',
                linewidth=0.5,
            )

    # Add reference lines
    ax.axhline(0, color='black', linestyle='--', alpha=0.4, linewidth=1.5, label='No fitness change')
    ax.axvline(0, color='black', linestyle='--', alpha=0.4, linewidth=1.5, label='No humanness change')

    # Labels and title
    ax.set_xlabel('Δ Humanness (IGLM log-likelihood)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Δ Fitness (Oracle score)', fontsize=14, fontweight='bold')
    ax.set_title('Fitness-Humanness Tradeoff: TAG Guidance Strength Sweep',
                 fontsize=16, fontweight='bold', pad=20)

    # Legend
    legend = ax.legend(loc='best', frameon=True, shadow=True, fontsize=11,
                      title='Guidance Strength (γ)', title_fontsize=12)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label='Guidance Strength (γ)')
    cbar.ax.tick_params(labelsize=10)

    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save
    scatter_plot = output_dir / "fitness_vs_humanness_scatter.png"
    plt.savefig(scatter_plot, dpi=300, bbox_inches='tight')
    print(f"Scatter plot saved to: {scatter_plot}")
    plt.close()


def create_summary_plots(df, guidance_strengths, output_dir):
    """Create summary boxplots for fitness and humanness deltas."""

    sns.set_style("whitegrid")
    plt.rcParams['font.size'] = 11

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Color palette - use string keys to match DataFrame
    colors = plt.cm.viridis(np.linspace(0, 1, len(guidance_strengths)))
    palette = {str(gamma): colors[i] for i, gamma in enumerate(guidance_strengths)}

    # Plot 1: Fitness delta boxplot
    plot_data_fitness = []
    for gamma in guidance_strengths:
        gamma_df = df[df['guidance_strength'] == gamma]
        for delta in gamma_df['fitness_delta']:
            plot_data_fitness.append({
                'Guidance Strength (γ)': str(gamma),
                'Δ Fitness': delta
            })

    plot_df_fitness = pd.DataFrame(plot_data_fitness)

    sns.boxplot(
        data=plot_df_fitness,
        x='Guidance Strength (γ)',
        y='Δ Fitness',
        palette=palette,
        ax=ax1,
        showfliers=False,
    )

    # Add individual points
    for i, gamma in enumerate(guidance_strengths):
        gamma_key = str(gamma)
        gamma_data = plot_df_fitness[plot_df_fitness['Guidance Strength (γ)'] == gamma_key]
        if len(gamma_data) > 0:
            y_vals = gamma_data['Δ Fitness'].values
            x_vals = np.random.normal(i, 0.04, size=len(y_vals))
            ax1.scatter(x_vals, y_vals, alpha=0.3, s=30, color=palette[gamma_key], zorder=10)

    ax1.axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    ax1.set_xlabel('Guidance Strength (γ)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Δ Fitness (Oracle score)', fontsize=13, fontweight='bold')
    ax1.set_title('Fitness Improvement by Guidance Strength', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    # Plot 2: Humanness delta boxplot
    plot_data_humanness = []
    for gamma in guidance_strengths:
        gamma_df = df[df['guidance_strength'] == gamma]
        for delta in gamma_df['humanness_delta']:
            plot_data_humanness.append({
                'Guidance Strength (γ)': str(gamma),
                'Δ Humanness': delta
            })

    plot_df_humanness = pd.DataFrame(plot_data_humanness)

    sns.boxplot(
        data=plot_df_humanness,
        x='Guidance Strength (γ)',
        y='Δ Humanness',
        palette=palette,
        ax=ax2,
        showfliers=False,
    )

    # Add individual points
    for i, gamma in enumerate(guidance_strengths):
        gamma_key = str(gamma)
        gamma_data = plot_df_humanness[plot_df_humanness['Guidance Strength (γ)'] == gamma_key]
        if len(gamma_data) > 0:
            y_vals = gamma_data['Δ Humanness'].values
            x_vals = np.random.normal(i, 0.04, size=len(y_vals))
            ax2.scatter(x_vals, y_vals, alpha=0.3, s=30, color=palette[gamma_key], zorder=10)

    ax2.axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    ax2.set_xlabel('Guidance Strength (γ)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Δ Humanness (IGLM log-likelihood)', fontsize=13, fontweight='bold')
    ax2.set_title('Humanness Change by Guidance Strength', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # Save
    summary_plot = output_dir / "summary_boxplots.png"
    plt.savefig(summary_plot, dpi=300, bbox_inches='tight')
    print(f"Summary plots saved to: {summary_plot}")
    plt.close()


if __name__ == "__main__":
    main()
