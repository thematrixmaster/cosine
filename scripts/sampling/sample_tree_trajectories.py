"""Sample protein sequences down phylogenetic trees using CTMC models with germline-rooted batched sampling.

This script implements batched trajectory sampling starting from germline-mapped oracle seeds.
Supports both unguided and TAG-guided Gillespie sampling on specified Rodriguez dataset trees.

Key features:
- Start from oracle seed sequences (binders only)
- Map seeds to germline using get_closest_germline or generate_naive_sequence
- True batched sampling: N independent trajectories down the tree
- Compare unguided vs TAG-guided methods

Usage:
    # Analyze trees first to find large trees
    uv run python scripts/analyze_rodriguez_trees.py

    # Sample 100 trajectories from a specific tree with germline mapping
    uv run python scripts/sample_tree_trajectories.py \
        --family-id 'sample-igg-SC-24_123' \
        --oracle-seed-idx 0 \
        --germline-method get_closest_germline \
        --batch-size 100 \
        --use-guided \
        --oracle SARSCoV2Beta
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
evo_path = project_root / 'evo'
sys.path.insert(0, str(evo_path))

import argparse
import json
import time
import random
from datetime import datetime
from typing import Dict, List, Optional, Callable
from collections import deque

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from tqdm import tqdm

from evo.tokenization import Vocab
from evo.phylogeny import df_to_ete3_tree
from evo.oracles import get_oracle
from evo.antibody import get_closest_germline, generate_naive_sequence
from cosine.models.modules.ctmc_module import CTMCModule
from cosine.models.nets.ctmc import NeuralCTMC, NeuralCTMCGenerator


def parse_args():
    parser = argparse.ArgumentParser(
        description='Sample protein sequences down phylogenetic trees with germline-rooted batched sampling'
    )

    # Data and model
    parser.add_argument('--dataset-path', type=str,
                        default='/accounts/projects/yss/stephen.lu/peint-workspace/main/data/dasm/rodriguez-airr-seq-race-prod-NoWinCheck_igh_pcp_2024-11-12_MASKED_NI_ConsCys_no-naive_DXSMVALID.csv',
                        help='Path to Rodriguez dataset CSV')
    parser.add_argument('--model-path', type=str,
                        default='/scratch/users/stephen.lu/projects/protevo/logs/train/runs/2026-01-06_18-32-49/checkpoints/epoch_001.ckpt',
                        help='Path to CTMC model checkpoint')

    # Tree and seed selection
    parser.add_argument('--family-id', type=str, required=True,
                        help='Family ID from Rodriguez dataset (e.g., "sample-igg-SC-24_123")')
    parser.add_argument('--oracle', type=str, default='SARSCoV2Beta',
                        choices=['SARSCoV1', 'SARSCoV2Beta'],
                        help='Oracle variant for seed selection (default: SARSCoV2Beta)')
    parser.add_argument('--oracle-seed-idx', type=int, default=0,
                        help='Index of oracle seed sequence to use (must be binder, default: 0)')

    # Germline mapping
    parser.add_argument('--germline-method', type=str, required=True,
                        choices=['get_closest_germline', 'generate_naive_sequence'],
                        help='Germline mapping method: get_closest_germline (preserves CDR3) or generate_naive_sequence (new CDR3)')
    parser.add_argument('--germline-seed', type=int, default=None,
                        help='Random seed for generate_naive_sequence method (optional, for reproducibility)')

    # Batch sampling
    parser.add_argument('--batch-size', type=int, default=100,
                        help='Number of independent trajectories to sample (default: 100)')
    parser.add_argument('--total-samples', type=int, default=None,
                        help='Total samples to generate; if > batch-size, will run multiple batches (default: None, uses batch-size)')

    # Guided sampling
    parser.add_argument('--use-guided', action='store_true',
                        help='Enable TAG guided sampling (in addition to unguided)')
    parser.add_argument('--guidance-strength', type=float, default=2.0,
                        help='Guidance strength γ for TAG (default: 2.0)')
    parser.add_argument('--oracle-chunk-size', type=int, default=5000,
                        help='Max sequences per oracle call (default: 5000)')

    # Sampling parameters
    parser.add_argument('--max-decode-steps', type=int, default=2048,
                        help='Maximum mutations per branch (default: 2048)')
    parser.add_argument('--n-sequences', type=int, default=5,
                        help='Sequences per branch for rejection sampling (default: 5)')

    # Output
    parser.add_argument('--output-dir', type=str,
                        default='/scratch/users/stephen.lu/projects/protevo/results/tree_sampling_batched',
                        help='Output directory for results')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility (default: None, uses timestamp)')

    return parser.parse_args()


def load_oracle_binders(oracle_variant: str) -> pd.DataFrame:
    """Load oracle seed sequences and filter to binders only.

    Parameters
    ----------
    oracle_variant : str
        'SARSCoV1' or 'SARSCoV2Beta'

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: sequence, binding (where binding=1)
    """
    csv_files = {
        "SARSCoV1": "CovAbDab_heavy_binds SARS-CoV1.csv",
        "SARSCoV2Beta": "CovAbDab_heavy_binds SARS-CoV2_Beta.csv",
    }

    # Get CSV path
    module_dir = Path(__file__).parent.parent / "evo" / "evo" / "oracles" / "data"
    csv_path = module_dir / csv_files[oracle_variant]

    if not csv_path.exists():
        raise FileNotFoundError(f"Oracle CSV not found: {csv_path}")

    # Load CSV
    df = pd.read_csv(csv_path)

    # Rename columns for clarity
    df.columns = ['sequence', 'binding']

    # Filter to binders only (binding=1)
    binders = df[df['binding'] == 1].reset_index(drop=True)

    print(f"Loaded {len(df)} total sequences from {csv_files[oracle_variant]}")
    print(f"Filtered to {len(binders)} binders (binding=1)")

    return binders


def select_and_map_oracle_seed(
    oracle_variant: str,
    seed_idx: int,
    germline_method: str,
    germline_seed: Optional[int] = None
) -> tuple[str, str, Dict]:
    """Select an oracle seed and map to germline.

    Parameters
    ----------
    oracle_variant : str
        Oracle variant name
    seed_idx : int
        Index of seed to use
    germline_method : str
        'get_closest_germline' or 'generate_naive_sequence'
    germline_seed : int, optional
        Random seed for generate_naive_sequence

    Returns
    -------
    tuple
        (original_seed, germline_root, metadata_dict)
    """
    # Load binders
    binders = load_oracle_binders(oracle_variant)

    # Check index is valid
    if seed_idx < 0 or seed_idx >= len(binders):
        raise ValueError(f"seed_idx {seed_idx} out of range [0, {len(binders)-1}]")

    # Get seed sequence
    original_seed = binders.iloc[seed_idx]['sequence']

    print(f"\n{'='*80}")
    print("ORACLE SEED SELECTION")
    print(f"{'='*80}")
    print(f"Oracle: {oracle_variant}")
    print(f"Seed index: {seed_idx}")
    print(f"Original sequence ({len(original_seed)} aa):")
    print(f"  {original_seed}")

    # Map to germline
    print(f"\nMapping to germline using: {germline_method}")

    if germline_method == 'get_closest_germline':
        germline_root = get_closest_germline(original_seed, scheme="imgt")
    elif germline_method == 'generate_naive_sequence':
        germline_root = generate_naive_sequence(original_seed, scheme="imgt", seed=germline_seed)
    else:
        raise ValueError(f"Unknown germline_method: {germline_method}")

    # Compare sequences
    n_changes = sum(1 for a, b in zip(original_seed, germline_root) if a != b)
    pct_similarity = (len(original_seed) - n_changes) / len(original_seed) * 100

    print(f"\nGermline-mapped sequence ({len(germline_root)} aa):")
    print(f"  {germline_root}")
    print(f"\nChanges: {n_changes} positions ({pct_similarity:.1f}% similarity)")

    # Create metadata
    metadata = {
        'oracle_variant': oracle_variant,
        'seed_idx': seed_idx,
        'original_seed': original_seed,
        'germline_root': germline_root,
        'germline_method': germline_method,
        'germline_seed': germline_seed,
        'n_changes': n_changes,
        'similarity_pct': pct_similarity
    }

    return original_seed, germline_root, metadata


def load_rodriguez_dataset(dataset_path: Path) -> pd.DataFrame:
    """Load Rodriguez dataset from CSV."""
    print(f"\nLoading Rodriguez dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)
    print(f"Loaded {len(df)} edges from dataset")
    return df


def build_tree_from_family(df: pd.DataFrame, family_id: str):
    """Build ete3 tree from family edges."""

    # Create family_id column if not exists
    if 'family_id' not in df.columns:
        df['family_id'] = df['sample_id'].astype(str) + '_' + df['family'].astype(str)

    # Filter edges for this family
    family_df = df[df['family_id'] == family_id].copy()

    if len(family_df) == 0:
        raise ValueError(f"No edges found for family_id: {family_id}")

    # Build tree using evo.phylogeny
    try:
        tree = df_to_ete3_tree(family_df)
        return tree, family_df
    except Exception as e:
        raise RuntimeError(f"Error building tree for {family_id}: {e}")


def get_leaf_sequences(tree, family_df: pd.DataFrame) -> Dict[str, str]:
    """Extract real sequences for leaf nodes."""

    leaf_seqs = {}

    for leaf in tree.get_leaves():
        leaf_name = leaf.name

        # Find edge where this node is the child
        edges = family_df[family_df['child_name'] == leaf_name]

        if len(edges) > 0:
            leaf_seqs[leaf_name] = edges.iloc[0]['child_heavy']

    return leaf_seqs


def simulate_evolution_batched(
    root_sequence: str,
    tree,
    batch_size: int,
    vocab: Vocab,
    generate_fn: Callable,
    device: str,
    n_sequences: int = 5,
    max_decode_steps: int = 2048,
    max_retries: int = 10,
    seed: Optional[int] = None,
) -> List[Dict[str, str]]:
    """Simulate evolution down a tree with batched independent trajectories.

    Parameters
    ----------
    root_sequence : str
        Root sequence (germline-mapped)
    tree : ete3.Tree
        Phylogenetic tree structure
    batch_size : int
        Number of independent trajectories to sample
    vocab : Vocab
        Tokenizer vocabulary
    generate_fn : Callable
        Generation function (unguided or guided)
    device : str
        Device for computation
    n_sequences : int
        Number of candidate sequences per rejection sampling attempt
    max_decode_steps : int
        Maximum mutations per branch
    max_retries : int
        Maximum retry attempts for rejection sampling
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    List[Dict[str, str]]
        List of trajectories, each trajectory is dict mapping node_name -> sequence
    """

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    # Initialize batch of trajectories
    # Each trajectory is a dict: node_name -> sequence
    trajectories = [{} for _ in range(batch_size)]

    # All trajectories start with same root
    root = tree.get_tree_root()
    for i in range(batch_size):
        trajectories[i][root.name] = root_sequence

    # Prepare root metadata
    root_length = len(root_sequence)
    root_length_with_special = root_length + int(vocab.prepend_bos) + int(vocab.append_eos)
    x_sizes = torch.tensor([[root_length_with_special]], device=device)

    # Length criterion for rejection sampling
    def length_criterion(seqs: list[str]) -> list[str]:
        return [s for s in seqs if len(s) == root_length]

    # BFS traversal
    queue = deque([root])

    pbar = tqdm(total=len([n for n in tree.traverse()]), desc="Sampling tree nodes")

    while queue:
        parent_node = queue.popleft()
        parent_name = parent_node.name

        children = parent_node.get_children()
        if len(children) == 0:
            pbar.update(1)
            continue

        # For each child, sample for all batch members
        for child in children:
            child_name = child.name
            branch_length = child.dist

            # Collect parent sequences from all trajectories
            parent_sequences = [trajectories[i][parent_name] for i in range(batch_size)]

            # Sample for each trajectory with rejection sampling
            for batch_idx in range(batch_size):
                parent_seq = parent_sequences[batch_idx]

                retry_count = 0
                sampled_seq = None

                while retry_count < max_retries:
                    # Encode parent sequence (adds BOS/EOS)
                    parent_encoded = vocab.encode_single_sequence(parent_seq)
                    x_rep = torch.from_numpy(parent_encoded).unsqueeze(0).repeat(n_sequences, 1).to(device)
                    t_rep = torch.tensor([branch_length] * n_sequences, device=device)

                    # Generate candidates
                    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        y_rep = generate_fn(
                            x=x_rep,
                            t=t_rep,
                            device=device,
                            p=None,
                            max_decode_steps=max_decode_steps,
                            x_sizes=x_sizes.repeat(n_sequences, 1)
                        )

                    # Convert to strings
                    candidates = []
                    for i in range(n_sequences):
                        seq_str = "".join([vocab.token(idx.item()) for idx in y_rep[i]
                                          if vocab.token(idx.item()) in set("ARNDCQEGHILKMFPSTWYV")])
                        candidates.append(seq_str)

                    # Apply length criterion
                    valid_candidates = length_criterion(candidates)

                    if len(valid_candidates) > 0:
                        # Randomly select one valid candidate
                        sampled_seq = random.choice(valid_candidates)
                        break

                    retry_count += 1

                if sampled_seq is None:
                    # Fallback: use parent sequence
                    sampled_seq = parent_seq

                # Store in trajectory
                trajectories[batch_idx][child_name] = sampled_seq

            # Add child to queue (only once, not per batch member)
            queue.append(child)

        pbar.update(1)

    pbar.close()

    return trajectories


def main():
    args = parse_args()

    # Set random seed
    if args.seed is not None:
        random_seed = args.seed
    else:
        random_seed = int(time.time())

    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    # Determine total samples
    if args.total_samples is None:
        total_samples = args.batch_size
    else:
        total_samples = args.total_samples

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{timestamp}_family_{args.family_id}_batch{args.batch_size}_seed{random_seed}"
    exp_dir = Path(args.output_dir) / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    print(f"Experiment directory: {exp_dir}")
    print(f"Random seed: {random_seed}\n")

    # Load model
    print(f"Loading CTMC model from: {args.model_path}")
    module = CTMCModule.load_from_checkpoint(str(args.model_path), map_location=device, strict=False)
    net: NeuralCTMC = module.net
    vocab: Vocab = net.vocab
    net = net.eval().to(device)
    generator = NeuralCTMCGenerator(neural_ctmc=net)

    # Select and map oracle seed to germline
    original_seed, germline_root, seed_metadata = select_and_map_oracle_seed(
        oracle_variant=args.oracle,
        seed_idx=args.oracle_seed_idx,
        germline_method=args.germline_method,
        germline_seed=args.germline_seed
    )

    # Load oracle if using guided sampling
    oracle = None
    if args.use_guided:
        print(f"\nLoading oracle ({args.oracle}) for guided sampling...")
        oracle = get_oracle(
            args.oracle,
            enable_mc_dropout=True,
            mc_samples=10,
            use_iglm_weighting=False,
            device=device,
        )

    # Load dataset and build tree
    dataset_path = Path(args.dataset_path)
    df = load_rodriguez_dataset(dataset_path)

    print(f"\n{'='*80}")
    print("TREE LOADING")
    print(f"{'='*80}")
    print(f"Family ID: {args.family_id}")

    tree, family_df = build_tree_from_family(df, args.family_id)
    leaf_seqs = get_leaf_sequences(tree, family_df)

    n_nodes = len([n for n in tree.traverse()])
    n_leaves = len(leaf_seqs)

    print(f"Tree size: {n_nodes} nodes, {n_leaves} leaves")
    print(f"Germline root length: {len(germline_root)} aa")

    # Print experiment summary
    print(f"\n{'='*80}")
    print("BATCHED TREE SAMPLING EXPERIMENT")
    print(f"{'='*80}")
    print(f"Family ID: {args.family_id}")
    print(f"Tree size: {n_nodes} nodes, {n_leaves} leaves")
    print(f"Batch size: {args.batch_size} independent trajectories")
    print(f"Methods: unguided" + (" + guided (TAG)" if args.use_guided else ""))
    if args.use_guided:
        print(f"Guidance strength: {args.guidance_strength}")
    print(f"Random seed: {random_seed}")
    print(f"{'='*80}\n")

    # Define sampling functions
    def unguided_generate_fn(x: Tensor, t: Tensor, device, p, max_decode_steps, x_sizes, **kwargs):
        return generator.generate_with_gillespie(
            t=t, x=x, x_sizes=x_sizes, max_decode_steps=max_decode_steps
        )

    def guided_generate_fn(x: Tensor, t: Tensor, device, p, max_decode_steps, x_sizes, **kwargs):
        return generator.generate_with_gillespie(
            t=t, x=x, x_sizes=x_sizes,
            oracle=oracle,
            guidance_strength=args.guidance_strength,
            use_taylor_approx=True,
            max_decode_steps=max_decode_steps,
            use_guidance=True,
            oracle_chunk_size=args.oracle_chunk_size,
        )

    # Storage for all results
    all_results = []

    # Sample with unguided method
    print(f"\n{'='*80}")
    print("UNGUIDED SAMPLING")
    print(f"{'='*80}")
    start_time = time.time()

    unguided_trajectories = simulate_evolution_batched(
        root_sequence=germline_root,
        tree=tree,
        batch_size=args.batch_size,
        vocab=vocab,
        generate_fn=unguided_generate_fn,
        device=device,
        n_sequences=args.n_sequences,
        max_decode_steps=args.max_decode_steps,
        seed=random_seed
    )

    unguided_time = time.time() - start_time
    print(f"\nUnguided sampling completed in {unguided_time:.1f}s")
    print(f"Generated {len(unguided_trajectories)} independent trajectories")

    # Store unguided results
    for batch_idx, trajectory in enumerate(unguided_trajectories):
        for node_name, seq in trajectory.items():
            # Get node metadata
            node = tree.search_nodes(name=node_name)[0]
            is_leaf = node.is_leaf()
            branch_length = node.dist if node.dist is not None else 0.0
            depth = node.get_distance(tree.get_tree_root())

            all_results.append({
                'family_id': args.family_id,
                'batch_idx': batch_idx,
                'node_name': node_name,
                'sequence': seq,
                'germline_root': germline_root,
                'original_oracle_seed': original_seed,
                'is_leaf': is_leaf,
                'branch_length': branch_length,
                'depth': depth,
                'method': 'unguided',
                'oracle_variant': args.oracle,
                'germline_method': args.germline_method,
                'sampling_time': unguided_time
            })

    # Sample with guided method if enabled
    if args.use_guided:
        print(f"\n{'='*80}")
        print("TAG-GUIDED SAMPLING")
        print(f"{'='*80}")
        start_time = time.time()

        guided_trajectories = simulate_evolution_batched(
            root_sequence=germline_root,
            tree=tree,
            batch_size=args.batch_size,
            vocab=vocab,
            generate_fn=guided_generate_fn,
            device=device,
            n_sequences=args.n_sequences,
            max_decode_steps=args.max_decode_steps,
            seed=random_seed + 10000  # Different seed for guided
        )

        guided_time = time.time() - start_time
        print(f"\nGuided sampling completed in {guided_time:.1f}s")
        print(f"Generated {len(guided_trajectories)} independent trajectories")

        # Store guided results
        for batch_idx, trajectory in enumerate(guided_trajectories):
            for node_name, seq in trajectory.items():
                # Get node metadata
                node = tree.search_nodes(name=node_name)[0]
                is_leaf = node.is_leaf()
                branch_length = node.dist if node.dist is not None else 0.0
                depth = node.get_distance(tree.get_tree_root())

                all_results.append({
                    'family_id': args.family_id,
                    'batch_idx': batch_idx,
                    'node_name': node_name,
                    'sequence': seq,
                    'germline_root': germline_root,
                    'original_oracle_seed': original_seed,
                    'is_leaf': is_leaf,
                    'branch_length': branch_length,
                    'depth': depth,
                    'method': 'guided_taylor',
                    'oracle_variant': args.oracle,
                    'germline_method': args.germline_method,
                    'sampling_time': guided_time
                })

    # Save results
    print(f"\n{'='*80}")
    print("SAVING RESULTS")
    print(f"{'='*80}")

    # Save samples CSV
    results_df = pd.DataFrame(all_results)
    samples_csv = exp_dir / "batched_samples.csv"
    results_df.to_csv(samples_csv, index=False)
    print(f"Batched samples saved to: {samples_csv}")
    print(f"Total rows: {len(results_df)}")

    # Save metadata JSON
    metadata = {
        'timestamp': timestamp,
        'family_id': args.family_id,
        'tree_size': n_nodes,
        'tree_leaves': n_leaves,
        'batch_size': args.batch_size,
        'total_trajectories': args.batch_size * (2 if args.use_guided else 1),
        'methods': ['unguided'] + (['guided_taylor'] if args.use_guided else []),
        'random_seed': random_seed,
        'oracle': args.oracle,
        'oracle_seed_idx': args.oracle_seed_idx,
        'germline_method': args.germline_method,
        'germline_seed': args.germline_seed,
        'original_oracle_seed': original_seed,
        'germline_root': germline_root,
        'n_germline_changes': seed_metadata['n_changes'],
        'germline_similarity_pct': seed_metadata['similarity_pct'],
        'guidance_strength': args.guidance_strength if args.use_guided else None,
        'model_path': str(args.model_path),
        'dataset_path': str(args.dataset_path)
    }

    metadata_json = exp_dir / "metadata.json"
    with open(metadata_json, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {metadata_json}")

    # Save tree structure JSON
    tree_structure = {
        'family_id': args.family_id,
        'n_nodes': n_nodes,
        'n_leaves': n_leaves,
        'root_name': tree.get_tree_root().name,
        'leaf_names': [leaf.name for leaf in tree.get_leaves()],
    }

    tree_json = exp_dir / "tree_structure.json"
    with open(tree_json, 'w') as f:
        json.dump(tree_structure, f, indent=2)
    print(f"Tree structure saved to: {tree_json}")

    # Print summary
    print(f"\n{'='*80}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*80}")
    print(f"Total samples: {len(results_df)}")
    print(f"Trajectories per method: {args.batch_size}")
    print(f"Methods: {results_df['method'].unique().tolist()}")
    print(f"\nAll results saved to: {exp_dir}")

    # Print analysis example
    print(f"\n{'='*80}")
    print("NEXT STEPS - EXAMPLE ANALYSIS")
    print(f"{'='*80}")
    print(f"\n# Load results in Python:")
    print(f"import pandas as pd")
    print(f"df = pd.read_csv('{samples_csv}')")
    print(f"\n# Get leaf sequences only:")
    print(f"leaves = df[df['is_leaf'] == True]")
    print(f"\n# Compare methods at leaves:")
    print(f"unguided_leaves = leaves[leaves['method'] == 'unguided']")
    if args.use_guided:
        print(f"guided_leaves = leaves[leaves['method'] == 'guided_taylor']")
    print(f"\n# Check trajectory independence:")
    print(f"# (Different batch_idx should have different sequences)")
    print(f"leaf_node = leaves['node_name'].iloc[0]")
    print(f"leaf_seqs = leaves[leaves['node_name'] == leaf_node]['sequence']")
    print(f"print(f'Unique sequences at leaf: {{len(leaf_seqs.unique())}}/{{len(leaf_seqs)}}')")


if __name__ == "__main__":
    main()
