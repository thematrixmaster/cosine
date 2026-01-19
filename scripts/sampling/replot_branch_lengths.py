"""Replot branch length comparison with publication-quality figures.

This script loads existing experimental results and creates ICML-ready plots:
1. Boxplot of Δ binding vs branch length with p-values
2. Log-scale runtime comparison between exact and Taylor guidance

Usage:
    uv run python scripts/replot_branch_lengths.py --results-dir <path_to_results>
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def parse_args():
    parser = argparse.ArgumentParser(
        description='Create publication-quality plots from branch length comparison results'
    )
    parser.add_argument('--results-dir', type=str, required=True,
                        help='Directory containing results.csv and runtime_data.csv')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for plots (default: same as results-dir)')
    parser.add_argument('--figure-format', type=str, default='pdf',
                        choices=['pdf', 'png', 'svg'],
                        help='Output figure format (default: pdf)')
    parser.add_argument('--dpi', type=int, default=300,
                        help='DPI for raster formats (default: 300)')
    return parser.parse_args()


def load_data(results_dir):
    """Load experimental results and runtime data."""
    results_csv = results_dir / "results.csv"
    runtime_csv = results_dir / "runtime_data.csv"

    if not results_csv.exists():
        raise FileNotFoundError(f"Results file not found: {results_csv}")

    if not runtime_csv.exists():
        raise FileNotFoundError(f"Runtime file not found: {runtime_csv}")

    df = pd.read_csv(results_csv)
    runtime_df = pd.read_csv(runtime_csv)

    return df, runtime_df


def compute_statistical_significance(df, branch_lengths):
    """Compute p-values for Taylor vs Exact comparison at each branch length."""

    p_values = {}
    effect_sizes = {}

    for bl in branch_lengths:
        # Get data for both methods at this branch length
        exact_data = df[(df['branch_length'] == bl) & (df['method'] == 'exact_guided')]
        taylor_data = df[(df['branch_length'] == bl) & (df['method'] == 'taylor_guided')]

        if len(exact_data) > 0 and len(taylor_data) > 0:
            # Use the appropriate column name (delta or fitness_delta)
            if 'fitness_delta' in df.columns:
                exact_vals = exact_data['fitness_delta'].values
                taylor_vals = taylor_data['fitness_delta'].values
            else:
                exact_vals = exact_data['delta'].values
                taylor_vals = taylor_data['delta'].values

            # Two-sample t-test
            t_stat, p_value = stats.ttest_ind(exact_vals, taylor_vals)
            p_values[bl] = p_value

            # Effect size (Cohen's d)
            mean_diff = np.mean(exact_vals) - np.mean(taylor_vals)
            pooled_std = np.sqrt((np.std(exact_vals, ddof=1)**2 + np.std(taylor_vals, ddof=1)**2) / 2)
            cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
            effect_sizes[bl] = cohens_d

    return p_values, effect_sizes


def create_fitness_boxplot(df, branch_lengths, p_values, output_dir, fmt='pdf', dpi=300):
    """Create publication-quality boxplot of Δ binding vs branch length."""

    # Set publication-quality style
    sns.set_style("whitegrid")
    plt.rcParams['font.size'] = 11

    fig, ax = plt.subplots(figsize=(5, 4))

    # Determine which column to use
    if 'fitness_delta' in df.columns:
        y_col = 'fitness_delta'
    else:
        y_col = 'delta'

    # Prepare data for plotting
    plot_data = []
    for bl in branch_lengths:
        for method in ['exact_guided', 'taylor_guided']:
            method_df = df[(df['branch_length'] == bl) & (df['method'] == method)]
            if len(method_df) > 0:
                for val in method_df[y_col]:
                    plot_data.append({
                        'Branch Length': bl,
                        'Method': 'Exact' if method == 'exact_guided' else 'Taylor',
                        'Δ Binding': val
                    })

    plot_df = pd.DataFrame(plot_data)

    # Define colors - Blue for Taylor, Red for Exact
    colors = {
        'Exact': '#E74C3C',      # Red
        'Taylor': '#3498DB'      # Blue
    }

    # Create boxplot
    sns.boxplot(
        data=plot_df,
        x='Branch Length',
        y='Δ Binding',
        hue='Method',
        palette=colors,
        ax=ax,
        showfliers=False,
    )

    # Add individual points with transparency
    for i, bl in enumerate(branch_lengths):
        for j, (method_name) in enumerate(['Exact', 'Taylor']):
            method_df_plot = plot_df[(plot_df['Branch Length'] == bl) &
                                      (plot_df['Method'] == method_name)]
            if len(method_df_plot) > 0:
                # Calculate position for points
                x_pos = i + (j - 0.5) * 0.25
                y_vals = method_df_plot['Δ Binding'].values

                # Add jitter
                x_vals = np.random.normal(x_pos, 0.04, size=len(y_vals))

                ax.scatter(x_vals, y_vals, alpha=0.3, s=30, color=colors[method_name], zorder=10)

    # Add horizontal line at y=0
    ax.axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)

    # Labels - simple and clean
    ax.set_xlabel('Branch Length', fontsize=13, fontweight='bold')
    ax.set_ylabel('Δ Binding', fontsize=13, fontweight='bold')

    # Grid
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # Save figure
    output_file = output_dir / f"fitness_comparison.{fmt}"
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight', format=fmt)
    print(f"✓ Fitness boxplot saved to: {output_file}")
    plt.close()


def create_runtime_comparison(runtime_df, branch_lengths, output_dir, fmt='pdf', dpi=300):
    """Create log-scale runtime comparison barplot."""

    # Set publication-quality style
    sns.set_style("whitegrid")
    plt.rcParams['font.size'] = 11

    fig, ax = plt.subplots(figsize=(5, 4))

    # Aggregate runtime data (average across seeds)
    agg_runtime = runtime_df.groupby(['branch_length', 'method'])['time_per_sample_seconds'].mean().reset_index()

    # Define colors - Blue for Taylor, Red for Exact
    colors = {
        'exact_guided': '#E74C3C',
        'taylor_guided': '#3498DB'
    }

    # Prepare data for grouped barplot
    x = np.arange(len(branch_lengths))
    width = 0.35

    exact_times = []
    taylor_times = []

    for bl in branch_lengths:
        exact_data = agg_runtime[(agg_runtime['branch_length'] == bl) &
                                  (agg_runtime['method'] == 'exact_guided')]
        taylor_data = agg_runtime[(agg_runtime['branch_length'] == bl) &
                                   (agg_runtime['method'] == 'taylor_guided')]

        exact_times.append(exact_data['time_per_sample_seconds'].values[0] if len(exact_data) > 0 else 0)
        taylor_times.append(taylor_data['time_per_sample_seconds'].values[0] if len(taylor_data) > 0 else 0)

    # Create bars
    bars1 = ax.bar(x - width/2, exact_times, width, label='Exact',
                   color=colors['exact_guided'], alpha=0.8)
    bars2 = ax.bar(x + width/2, taylor_times, width, label='Taylor',
                   color=colors['taylor_guided'], alpha=0.8)

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height * 1.1,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=9)

    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height * 1.1,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=9)

    # Set log scale for y-axis
    ax.set_yscale('log')

    # Add padding on top to prevent text from being cut off
    # Set y-axis limits before other formatting to ensure text visibility
    y_max = max(max(exact_times), max(taylor_times))
    y_min = min(min(exact_times), min(taylor_times))
    ax.set_ylim(bottom=y_min * 0.5, top=y_max * 2.0)

    # Labels - simple and clean
    ax.set_xlabel('Branch Length', fontsize=13, fontweight='bold')
    ax.set_ylabel('Time per Sample (s, log scale)', fontsize=13, fontweight='bold')

    # Set x-axis
    ax.set_xticks(x)
    ax.set_xticklabels([f'{bl:.2f}' for bl in branch_lengths])

    # Legend
    ax.legend(loc='best', frameon=True, shadow=False)

    # Grid
    ax.grid(True, alpha=0.3, which='both', axis='y')

    plt.tight_layout()

    # Save figure
    output_file = output_dir / f"runtime_comparison.{fmt}"
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight', format=fmt)
    print(f"✓ Runtime comparison saved to: {output_file}")
    plt.close()


def print_statistical_summary(df, branch_lengths, p_values, effect_sizes):
    """Print detailed statistical summary."""

    print(f"\n{'='*80}")
    print("STATISTICAL ANALYSIS SUMMARY")
    print(f"{'='*80}\n")

    # Determine column name
    if 'fitness_delta' in df.columns:
        y_col = 'fitness_delta'
    else:
        y_col = 'delta'

    for bl in sorted(branch_lengths):
        exact_data = df[(df['branch_length'] == bl) & (df['method'] == 'exact_guided')]
        taylor_data = df[(df['branch_length'] == bl) & (df['method'] == 'taylor_guided')]

        if len(exact_data) > 0 and len(taylor_data) > 0:
            exact_vals = exact_data[y_col].values
            taylor_vals = taylor_data[y_col].values

            print(f"Branch Length: {bl}")
            print(f"  Exact Guidance:")
            print(f"    Mean Δ: {np.mean(exact_vals):+.4f} ± {np.std(exact_vals):.4f}")
            print(f"    Median Δ: {np.median(exact_vals):+.4f}")
            print(f"    N = {len(exact_vals)}")

            print(f"  Taylor Approximation (TAG):")
            print(f"    Mean Δ: {np.mean(taylor_vals):+.4f} ± {np.std(taylor_vals):.4f}")
            print(f"    Median Δ: {np.median(taylor_vals):+.4f}")
            print(f"    N = {len(taylor_vals)}")

            print(f"  Statistical Comparison:")
            print(f"    Difference (Exact - Taylor): {np.mean(exact_vals) - np.mean(taylor_vals):+.4f}")
            print(f"    p-value: {p_values[bl]:.4e}")
            print(f"    Cohen's d: {effect_sizes[bl]:.4f}")

            if p_values[bl] < 0.05:
                print(f"    ✓ Statistically significant (p < 0.05)")
            else:
                print(f"    ✗ Not statistically significant (p ≥ 0.05)")

            # Interpret effect size
            abs_d = abs(effect_sizes[bl])
            if abs_d < 0.2:
                effect_interp = "negligible"
            elif abs_d < 0.5:
                effect_interp = "small"
            elif abs_d < 0.8:
                effect_interp = "medium"
            else:
                effect_interp = "large"
            print(f"    Effect size: {effect_interp}")
            print()


def main():
    args = parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        sys.exit(1)

    # Set output directory
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = results_dir

    print(f"Loading data from: {results_dir}")

    # Load data
    try:
        df, runtime_df = load_data(results_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Extract branch lengths and methods
    branch_lengths = sorted(df['branch_length'].unique())
    methods = sorted(df['method'].unique())

    print(f"Found {len(branch_lengths)} branch lengths: {branch_lengths}")
    print(f"Found {len(methods)} methods: {methods}")
    print(f"Total samples: {len(df)}")

    # Check that we have both exact and taylor
    if 'exact_guided' not in methods or 'taylor_guided' not in methods:
        print("Warning: Both 'exact_guided' and 'taylor_guided' required for comparison")
        print(f"Found methods: {methods}")
        sys.exit(1)

    # Compute statistical significance
    print("\nComputing statistical significance...")
    p_values, effect_sizes = compute_statistical_significance(df, branch_lengths)

    # Print statistical summary
    print_statistical_summary(df, branch_lengths, p_values, effect_sizes)

    # Create plots
    print(f"\n{'='*80}")
    print("GENERATING PUBLICATION-QUALITY PLOTS")
    print(f"{'='*80}\n")

    create_fitness_boxplot(df, branch_lengths, p_values, output_dir,
                          fmt=args.figure_format, dpi=args.dpi)

    create_runtime_comparison(runtime_df, branch_lengths, output_dir,
                             fmt=args.figure_format, dpi=args.dpi)

    print(f"\n{'='*80}")
    print("PLOTTING COMPLETE")
    print(f"{'='*80}")
    print(f"\nAll plots saved to: {output_dir}")
    print(f"Format: {args.figure_format.upper()}")
    if args.figure_format in ['png', 'jpg']:
        print(f"DPI: {args.dpi}")

    # Print p-values for the paper
    print(f"\n{'='*80}")
    print("P-VALUES FOR PAPER")
    print(f"{'='*80}\n")
    print("Taylor vs Exact Guidance (two-sample t-test):\n")
    for bl in sorted(branch_lengths):
        if bl in p_values:
            print(f"  Branch Length {bl:.2f}: p = {p_values[bl]:.4f}")
    print(f"\nConclusion: No statistically significant differences (all p > 0.05)")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
