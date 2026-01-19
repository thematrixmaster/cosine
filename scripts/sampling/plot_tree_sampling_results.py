"""Plot comparison of tree sampling results.

This script analyzes tree sampling results and creates publication-quality
boxplots comparing oracle fitness and humanness scores across:
1. Real Leaves (ground truth from Rodriguez test set)
2. Unguided Leaves (sampled without guidance)
3. Guided Leaves (sampled with TAG guidance)
4. Real Binders (known SARS-CoV1 binders from CovAbDab)

Usage:
    uv run python scripts/plot_tree_sampling_results.py \\
        --results-dir /path/to/tree_sampling/results \\
        --figure-format pdf
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
import torch

from evo.oracles import get_oracle
from evo.antibody import compute_oasis_humanness


def parse_args():
    parser = argparse.ArgumentParser(
        description='Plot tree sampling results comparison'
    )
    parser.add_argument('--results-dir', type=str, required=True,
                        help='Path to tree sampling results directory')
    parser.add_argument('--oracle', type=str, default='SARSCoV1',
                        choices=['SARSCoV1', 'SARSCoV2Beta'],
                        help='Oracle to use for fitness scoring (default: SARSCoV1)')
    parser.add_argument('--binders-csv', type=str,
                        default='/accounts/projects/yss/stephen.lu/peint-workspace/main/evo/evo/oracles/data/CovAbDab_heavy_binds SARS-CoV1.csv',
                        help='Path to binders CSV file')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for plots (default: same as results-dir)')
    parser.add_argument('--figure-format', type=str, default='pdf',
                        choices=['pdf', 'png', 'svg'],
                        help='Output figure format (default: pdf)')
    parser.add_argument('--dpi', type=int, default=300,
                        help='DPI for raster formats (default: 300)')
    return parser.parse_args()


def load_tree_sampling_results(results_dir: Path) -> pd.DataFrame:
    """Load tree sampling results from CSV."""
    results_csv = results_dir / "all_samples.csv"

    if not results_csv.exists():
        raise FileNotFoundError(f"Results file not found: {results_csv}")

    print(f"Loading results from: {results_csv}")
    df = pd.read_csv(results_csv)
    print(f"Loaded {len(df)} samples")

    return df


def load_real_binders(binders_csv: Path) -> list:
    """Load real binder sequences from CovAbDab CSV."""
    print(f"Loading real binders from: {binders_csv}")

    binder_df = pd.read_csv(binders_csv)

    # Filter for sequences that bind SARS-CoV1
    binders = binder_df[binder_df['binds SARS-CoV1'] == 1]['VHorVHH'].tolist()

    print(f"Loaded {len(binders)} real binder sequences")

    return binders


def extract_sequences_by_category(df: pd.DataFrame, real_binders: list) -> dict:
    """Extract sequences for each category."""

    categories = {}

    # 1. Real Leaves - ground truth from test set
    real_leaves_df = df[df['is_leaf'] == True].copy()
    real_leaves_df = real_leaves_df[real_leaves_df['real_sequence'].notna()]
    real_leaves_df = real_leaves_df[real_leaves_df['real_sequence'] != '']

    # Get unique real sequences (same across replicates)
    real_leaf_seqs = real_leaves_df['real_sequence'].unique().tolist()
    categories['Real Leaves'] = real_leaf_seqs
    print(f"Real Leaves: {len(real_leaf_seqs)} unique sequences")

    # 2. Unguided Leaves - sampled without guidance
    unguided_leaves = df[(df['method'] == 'unguided') & (df['is_leaf'] == True)]
    unguided_seqs = unguided_leaves['sequence'].tolist()
    categories['Unguided Leaves'] = unguided_seqs
    print(f"Unguided Leaves: {len(unguided_seqs)} sequences")

    # 3. Guided Leaves - sampled with TAG guidance
    guided_leaves = df[(df['method'] == 'guided') & (df['is_leaf'] == True)]
    guided_seqs = guided_leaves['sequence'].tolist()
    categories['Guided Leaves'] = guided_seqs
    print(f"Guided Leaves: {len(guided_seqs)} sequences")

    # 4. Real Binders - known binders from CovAbDab
    categories['Real Binders'] = real_binders
    print(f"Real Binders: {len(real_binders)} sequences")

    return categories


def score_sequences(sequences_by_category: dict, oracle_name: str, device: str) -> tuple:
    """Score all sequences with oracle and humanness."""

    print(f"\n{'='*80}")
    print("SCORING SEQUENCES")
    print(f"{'='*80}\n")

    # Load oracle
    print(f"Loading {oracle_name} oracle...")
    oracle = get_oracle(
        oracle_name,
        enable_mc_dropout=True,
        mc_samples=10,
        use_iglm_weighting=False,
        device=device,
    )

    fitness_scores = {}
    humanness_scores = {}

    for category, sequences in sequences_by_category.items():
        print(f"\nScoring {category}...")

        # Oracle fitness scoring
        print(f"  Computing oracle fitness...")
        scores, _ = oracle.predict_batch(sequences)
        fitness_scores[category] = scores
        print(f"  Mean fitness: {np.mean(scores):.4f} ± {np.std(scores):.4f}")

        # Humanness scoring
        print(f"  Computing OASIS humanness...")
        humanness = compute_oasis_humanness(sequences, chain="heavy")
        humanness_scores[category] = np.array(humanness)
        print(f"  Mean humanness: {np.mean(humanness):.4f} ± {np.std(humanness):.4f}")

    return fitness_scores, humanness_scores


def create_comparison_plots(
    fitness_scores: dict,
    humanness_scores: dict,
    output_dir: Path,
    fmt: str = 'pdf',
    dpi: int = 300
):
    """Create side-by-side boxplots for fitness and humanness."""

    # Set style matching sweep_tag_guidance.py
    sns.set_style("whitegrid")
    plt.rcParams['font.size'] = 11

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Color palette using viridis
    categories = list(fitness_scores.keys())
    colors_array = plt.cm.viridis(np.linspace(0, 1, len(categories)))
    palette = {cat: colors_array[i] for i, cat in enumerate(categories)}

    # ========== Plot 1: Fitness Boxplot ==========

    # Prepare data for fitness plot
    fitness_data = []
    for category, scores in fitness_scores.items():
        for score in scores:
            fitness_data.append({
                'Category': category,
                'Fitness': score
            })

    fitness_df = pd.DataFrame(fitness_data)

    # Create boxplot
    sns.boxplot(
        data=fitness_df,
        x='Category',
        y='Fitness',
        palette=palette,
        ax=ax1,
        showfliers=False,
    )

    # Add individual points with jitter
    for i, category in enumerate(categories):
        cat_data = fitness_df[fitness_df['Category'] == category]
        y_vals = cat_data['Fitness'].values
        x_vals = np.random.normal(i, 0.04, size=len(y_vals))
        ax1.scatter(x_vals, y_vals, alpha=0.3, s=30, color=palette[category], zorder=10)

    # Formatting
    ax1.axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    ax1.set_xlabel('Category', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Oracle Fitness Score', fontsize=13, fontweight='bold')
    ax1.set_title('Binding Affinity Comparison', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.tick_params(axis='x', rotation=15)

    # ========== Plot 2: Humanness Boxplot ==========

    # Prepare data for humanness plot
    humanness_data = []
    for category, scores in humanness_scores.items():
        for score in scores:
            humanness_data.append({
                'Category': category,
                'Humanness': score
            })

    humanness_df = pd.DataFrame(humanness_data)

    # Create boxplot
    sns.boxplot(
        data=humanness_df,
        x='Category',
        y='Humanness',
        palette=palette,
        ax=ax2,
        showfliers=False,
    )

    # Add individual points with jitter
    for i, category in enumerate(categories):
        cat_data = humanness_df[humanness_df['Category'] == category]
        y_vals = cat_data['Humanness'].values
        x_vals = np.random.normal(i, 0.04, size=len(y_vals))
        ax2.scatter(x_vals, y_vals, alpha=0.3, s=30, color=palette[category], zorder=10)

    # Formatting
    ax2.axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    ax2.set_xlabel('Category', fontsize=13, fontweight='bold')
    ax2.set_ylabel('OASIS Humanness Score', fontsize=13, fontweight='bold')
    ax2.set_title('Humanness Comparison', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.tick_params(axis='x', rotation=15)

    # Save figure
    plt.tight_layout()
    output_file = output_dir / f"fitness_humanness_comparison.{fmt}"
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight', format=fmt)
    print(f"\n✓ Comparison plot saved to: {output_file}")
    plt.close()


def save_summary_statistics(
    fitness_scores: dict,
    humanness_scores: dict,
    output_dir: Path
):
    """Save summary statistics to CSV."""

    summary_data = []

    for category in fitness_scores.keys():
        fit_scores = fitness_scores[category]
        human_scores = humanness_scores[category]

        summary_data.append({
            'category': category,
            'n_sequences': len(fit_scores),
            'fitness_mean': np.mean(fit_scores),
            'fitness_std': np.std(fit_scores),
            'fitness_median': np.median(fit_scores),
            'fitness_min': np.min(fit_scores),
            'fitness_max': np.max(fit_scores),
            'humanness_mean': np.mean(human_scores),
            'humanness_std': np.std(human_scores),
            'humanness_median': np.median(human_scores),
            'humanness_min': np.min(human_scores),
            'humanness_max': np.max(human_scores),
        })

    summary_df = pd.DataFrame(summary_data)
    summary_csv = output_dir / "summary_statistics.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"✓ Summary statistics saved to: {summary_csv}")

    # Print summary table
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}\n")
    print(summary_df.to_string(index=False))


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    # Set paths
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        sys.exit(1)

    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = results_dir

    binders_csv = Path(args.binders_csv)
    if not binders_csv.exists():
        print(f"Error: Binders CSV not found: {binders_csv}")
        sys.exit(1)

    print(f"{'='*80}")
    print("TREE SAMPLING ANALYSIS")
    print(f"{'='*80}\n")

    # Load data
    print("Loading data...")
    df = load_tree_sampling_results(results_dir)
    real_binders = load_real_binders(binders_csv)

    # Extract sequences by category
    print(f"\n{'='*80}")
    print("EXTRACTING SEQUENCES BY CATEGORY")
    print(f"{'='*80}\n")
    sequences_by_category = extract_sequences_by_category(df, real_binders)

    # Score sequences
    fitness_scores, humanness_scores = score_sequences(
        sequences_by_category,
        args.oracle,
        device
    )

    # Create plots
    print(f"\n{'='*80}")
    print("CREATING PLOTS")
    print(f"{'='*80}")

    create_comparison_plots(
        fitness_scores,
        humanness_scores,
        output_dir,
        fmt=args.figure_format,
        dpi=args.dpi
    )

    # Save summary statistics
    save_summary_statistics(fitness_scores, humanness_scores, output_dir)

    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"\nAll results saved to: {output_dir}")


if __name__ == "__main__":
    main()
