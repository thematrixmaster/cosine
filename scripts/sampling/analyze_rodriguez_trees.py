"""Analyze Rodriguez dataset to identify large phylogenetic trees.

This script helps identify suitable trees for tree sampling experiments by
computing statistics on tree size, depth, and branch lengths.

Usage:
    uv run python scripts/analyze_rodriguez_trees.py
    uv run python scripts/analyze_rodriguez_trees.py --output-csv tree_stats.csv
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import pandas as pd
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(
        description='Analyze Rodriguez dataset trees'
    )
    parser.add_argument('--dataset-path', type=str,
                        default='/accounts/projects/yss/stephen.lu/peint-workspace/main/data/dasm/rodriguez-airr-seq-race-prod-NoWinCheck_igh_pcp_2024-11-12_MASKED_NI_ConsCys_no-naive_DXSMVALID.csv',
                        help='Path to Rodriguez dataset CSV')
    parser.add_argument('--output-csv', type=str,
                        default=None,
                        help='Output CSV path for tree statistics (optional)')
    parser.add_argument('--top-n', type=int, default=20,
                        help='Number of largest trees to display (default: 20)')

    return parser.parse_args()


def analyze_trees(df: pd.DataFrame):
    """Compute tree statistics for each family."""

    # Create family IDs
    df['family_id'] = df['sample_id'].astype(str) + '_' + df['family'].astype(str)

    # Group by family
    families = df.groupby('family_id')

    tree_stats = []

    for family_id, family_df in families:
        # Count nodes (unique parent + unique child names)
        all_nodes = set(family_df['parent_name'].tolist() + family_df['child_name'].tolist())
        n_nodes = len(all_nodes)

        # Count edges
        n_edges = len(family_df)

        # Count leaves (children that never appear as parents)
        parents = set(family_df['parent_name'].unique())
        children = set(family_df['child_name'].unique())
        leaves = children - parents
        n_leaves = len(leaves)

        # Max depth
        max_depth = family_df['depth'].max() if 'depth' in family_df.columns else None

        # Total branch length
        total_branch_length = family_df['branch_length'].sum()
        mean_branch_length = family_df['branch_length'].mean()

        # Sample ID
        sample_id = family_df['sample_id'].iloc[0]
        family_num = family_df['family'].iloc[0]

        tree_stats.append({
            'family_id': family_id,
            'sample_id': sample_id,
            'family': family_num,
            'n_nodes': n_nodes,
            'n_edges': n_edges,
            'n_leaves': n_leaves,
            'max_depth': max_depth,
            'total_branch_length': total_branch_length,
            'mean_branch_length': mean_branch_length,
        })

    return pd.DataFrame(tree_stats)


def main():
    args = parse_args()

    print("="*80)
    print("RODRIGUEZ DATASET TREE ANALYSIS")
    print("="*80)

    # Load dataset
    print(f"\nLoading dataset from: {args.dataset_path}")
    df = pd.read_csv(args.dataset_path)
    print(f"Loaded {len(df)} edges")

    # Analyze trees
    print("\nAnalyzing trees...")
    tree_stats = analyze_trees(df)

    # Sort by number of nodes (descending)
    tree_stats = tree_stats.sort_values('n_nodes', ascending=False).reset_index(drop=True)

    print(f"\nFound {len(tree_stats)} unique trees/families")
    print(f"\nTree size statistics:")
    print(f"  Mean nodes: {tree_stats['n_nodes'].mean():.1f}")
    print(f"  Median nodes: {tree_stats['n_nodes'].median():.1f}")
    print(f"  Min nodes: {tree_stats['n_nodes'].min()}")
    print(f"  Max nodes: {tree_stats['n_nodes'].max()}")

    # Display top N largest trees
    print(f"\n{'='*80}")
    print(f"TOP {args.top_n} LARGEST TREES")
    print(f"{'='*80}")
    print("\nTo use a tree in sampling, copy the family_id and use it with:")
    print("  python scripts/sample_tree_trajectories.py --family-id <family_id> ...\n")

    # Format output
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 120)
    pd.set_option('display.max_colwidth', 30)

    top_trees = tree_stats.head(args.top_n)
    print(top_trees.to_string(index=True))

    # Save to CSV if requested
    if args.output_csv:
        output_path = Path(args.output_csv)
        tree_stats.to_csv(output_path, index=False)
        print(f"\n{'='*80}")
        print(f"Full tree statistics saved to: {output_path}")

    # Print example usage
    example_family = top_trees.iloc[0]['family_id']
    print(f"\n{'='*80}")
    print("EXAMPLE USAGE")
    print(f"{'='*80}")
    print(f"\n# Sample from the largest tree ({top_trees.iloc[0]['n_nodes']} nodes):")
    print(f"python scripts/sample_tree_trajectories.py \\")
    print(f"  --family-id '{example_family}' \\")
    print(f"  --oracle-seed-idx 0 \\")
    print(f"  --germline-method get_closest_germline \\")
    print(f"  --batch-size 100")


if __name__ == "__main__":
    main()
