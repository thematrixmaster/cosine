"""Regenerate TAG guidance sweep plots with OASIS humanness scores.

ICML-style publication-ready plots with blue gradient for guidance strengths.
Creates three separate PDF plots.
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from evo.antibody import compute_oasis_humanness


def parse_args():
    parser = argparse.ArgumentParser(
        description='Regenerate plots with OASIS humanness scores'
    )
    parser.add_argument('--results-dir', type=str, required=True,
                        help='Directory containing results.csv')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: same as results-dir)')
    return parser.parse_args()


def main():
    args = parse_args()

    results_dir = Path(args.results_dir)
    results_csv = results_dir / "results.csv"

    if not results_csv.exists():
        print(f"Error: Results file not found: {results_csv}")
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else results_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from: {results_csv}")
    df = pd.read_csv(results_csv)

    # Get unique sequences (seeds + sampled)
    all_seqs = list(df['seed_seq'].unique()) + list(df['sampled_seq'].unique())
    unique_seqs = list(set(all_seqs))

    print(f"\nComputing OASIS humanness for {len(unique_seqs)} unique sequences...")
    oasis_scores = compute_oasis_humanness(unique_seqs)

    # Create mapping
    seq_to_oasis = dict(zip(unique_seqs, oasis_scores))

    # Add OASIS scores to dataframe
    print("Adding OASIS scores to dataframe...")
    df['seed_oasis'] = df['seed_seq'].map(seq_to_oasis)
    df['sampled_oasis'] = df['sampled_seq'].map(seq_to_oasis)
    df['oasis_delta'] = df['sampled_oasis'] - df['seed_oasis']

    # Save updated results
    updated_csv = output_dir / "results_with_oasis.csv"
    df.to_csv(updated_csv, index=False)
    print(f"Updated results saved to: {updated_csv}")

    # Extract guidance strengths
    guidance_strengths = sorted(df['guidance_strength'].unique())

    print(f"\nGenerating ICML-style plots for {len(guidance_strengths)} guidance strengths...")
    create_icml_plots(df, guidance_strengths, output_dir)

    print(f"\nAll plots saved to: {output_dir}")


def create_icml_plots(df, guidance_strengths, output_dir):
    """Create three separate ICML-style publication-ready boxplots."""

    # Slightly narrower width for better aesthetics
    figsize = (3.0, 2.5)

    # Blue gradient palette - colorblind-friendly
    n_colors = len(guidance_strengths)
    blues = plt.cm.Blues(np.linspace(0.5, 0.95, n_colors))

    positions = np.arange(len(guidance_strengths))

    # Compute mean seed OASIS percentile for reference line
    mean_seed_oasis = df['seed_oasis'].mean() * 100  # Convert to percentile

    # ========================================================================
    # PLOT 1: Fitness Delta
    # ========================================================================
    plt.figure(figsize=figsize)

    plot_data_fitness = []
    for gamma in guidance_strengths:
        gamma_df = df[df['guidance_strength'] == gamma]
        for delta in gamma_df['fitness_delta']:
            plot_data_fitness.append({
                'γ': gamma,
                'Δ Fitness': delta
            })

    plot_df_fitness = pd.DataFrame(plot_data_fitness)

    # Create boxplot
    bp1 = plt.boxplot(
        [plot_df_fitness[plot_df_fitness['γ'] == g]['Δ Fitness'].values
         for g in guidance_strengths],
        positions=positions,
        widths=0.6,
        patch_artist=True,
        showfliers=False,
        boxprops=dict(linewidth=1.5),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
        medianprops=dict(linewidth=1.5, color='darkred'),
    )

    # Color boxes with blue gradient
    for patch, color in zip(bp1['boxes'], blues):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    plt.xlabel('Guidance Strength', fontsize=10)
    plt.ylabel('Δ Fitness', fontsize=10)
    plt.grid(True, alpha=0.2, linewidth=0.5)
    plt.xticks(positions, [f'{g:.1f}' for g in guidance_strengths], fontsize=9)
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
    # PLOT 2: Absolute OASIS Percentile (not delta)
    # ========================================================================
    plt.figure(figsize=figsize)

    plot_data_oasis = []
    for gamma in guidance_strengths:
        gamma_df = df[df['guidance_strength'] == gamma]
        for oasis in gamma_df['sampled_oasis']:
            plot_data_oasis.append({
                'γ': gamma,
                'OASIS': oasis * 100  # Convert to percentile
            })

    plot_df_oasis = pd.DataFrame(plot_data_oasis)

    # Create boxplot
    bp2 = plt.boxplot(
        [plot_df_oasis[plot_df_oasis['γ'] == g]['OASIS'].values
         for g in guidance_strengths],
        positions=positions,
        widths=0.6,
        patch_artist=True,
        showfliers=False,
        boxprops=dict(linewidth=1.5),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
        medianprops=dict(linewidth=1.5, color='darkred'),
    )

    # Color boxes with blue gradient
    for patch, color in zip(bp2['boxes'], blues):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Add horizontal line for mean seed OASIS percentile
    plt.axhline(mean_seed_oasis, color='black', linestyle='--',
                linewidth=1.0, alpha=0.5, label=f'Seed mean ({mean_seed_oasis:.1f}%)')

    plt.xlabel('Guidance Strength', fontsize=10)
    plt.ylabel('OASIS Percentile', fontsize=10)
    plt.legend(fontsize=7, loc='best', frameon=True, fancybox=False, edgecolor='black')
    plt.grid(True, alpha=0.2, linewidth=0.5)
    plt.xticks(positions, [f'{g:.1f}' for g in guidance_strengths], fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout(pad=0.3)

    # Save
    plot_path = output_dir / 'oasis_percentile.pdf'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    png_path = output_dir / 'oasis_percentile.png'
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    print(f"Saved {plot_path} and {png_path}")
    plt.close()

    # ========================================================================
    # PLOT 3: Number of Mutations
    # ========================================================================
    plt.figure(figsize=figsize)

    plot_data_mutations = []
    for gamma in guidance_strengths:
        gamma_df = df[df['guidance_strength'] == gamma]
        for n_mut in gamma_df['num_mutations']:
            plot_data_mutations.append({
                'γ': gamma,
                'Mutations': n_mut
            })

    plot_df_mutations = pd.DataFrame(plot_data_mutations)

    # Create boxplot
    bp3 = plt.boxplot(
        [plot_df_mutations[plot_df_mutations['γ'] == g]['Mutations'].values
         for g in guidance_strengths],
        positions=positions,
        widths=0.6,
        patch_artist=True,
        showfliers=False,
        boxprops=dict(linewidth=1.5),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
        medianprops=dict(linewidth=1.5, color='darkred'),
    )

    # Color boxes with blue gradient
    for patch, color in zip(bp3['boxes'], blues):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    plt.xlabel('Guidance Strength', fontsize=10)
    plt.ylabel('Number of Mutations', fontsize=10)
    plt.grid(True, alpha=0.2, linewidth=0.5)
    plt.xticks(positions, [f'{g:.1f}' for g in guidance_strengths], fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout(pad=0.3)

    # Save
    plot_path = output_dir / 'num_mutations.pdf'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    png_path = output_dir / 'num_mutations.png'
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    print(f"Saved {plot_path} and {png_path}")
    plt.close()

    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"\nSeed OASIS Percentile (mean): {mean_seed_oasis:.2f}%")
    print(f"{'='*80}")
    for gamma in guidance_strengths:
        gamma_df = df[df['guidance_strength'] == gamma]
        print(f"\nγ = {gamma}:")
        print(f"  Δ Fitness: {gamma_df['fitness_delta'].mean():+.4f} ± {gamma_df['fitness_delta'].std():.4f}")
        print(f"  OASIS Percentile: {gamma_df['sampled_oasis'].mean()*100:.2f} ± {gamma_df['sampled_oasis'].std()*100:.2f}")
        print(f"  Num Mutations: {gamma_df['num_mutations'].mean():.2f} ± {gamma_df['num_mutations'].std():.2f}")


if __name__ == "__main__":
    main()
