"""Sample protein sequences with fixed branch length using CTMC models.

This is a simplified version that:
- Starts from oracle seed antibody sequences (no germline mapping)
- Samples N independent trajectories with a single fixed branch length
- Supports both unguided and TAG-guided sampling
- Includes mutation ceiling feature via post-hoc rejection sampling
- Supports region masking (e.g., CDR-only mutations)

Usage:
    # Sample 100 trajectories with fixed branch length, no mutation ceiling
    uv run python scripts/sample_fixed_branch.py \
        --oracle SARSCoV2Beta \
        --oracle-seed-idx 0 \
        --branch-length 0.5 \
        --batch-size 100 \
        --use-guided \
        --guidance-strength 2.0

    # With mutation ceiling
    uv run python scripts/sample_fixed_branch.py \
        --oracle SARSCoV2Beta \
        --oracle-seed-idx 0 \
        --branch-length 0.5 \
        --max-mutations 20 \
        --batch-size 100 \
        --use-guided

    # With CDR-only mutations
    uv run python scripts/sample_fixed_branch.py \
        --oracle SARSCoV2Beta \
        --oracle-seed-idx 0 \
        --branch-length 0.5 \
        --max-mutations 20 \
        --mask-region CDR_overall \
        --batch-size 100
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
evo_path = project_root / 'evo'
sys.path.insert(0, str(evo_path))

import argparse
import time
import random
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from tqdm import tqdm

from evo.tokenization import Vocab
from evo.oracles import get_oracle
from evo.antibody import create_region_masks, compute_oasis_humanness, compute_iglm_humanness
from peint.models.modules.ctmc_module import CTMCModule
from peint.models.nets.ctmc import NeuralCTMC, NeuralCTMCGenerator
from oracle_counter import OracleCallCounter


def parse_args():
    parser = argparse.ArgumentParser(
        description='Sample protein sequences with fixed branch length from oracle seeds'
    )

    # Model
    parser.add_argument('--model-path', type=str,
                        default='/scratch/users/stephen.lu/projects/protevo/logs/train/runs/2026-01-06_18-32-49/checkpoints/epoch_001.ckpt',  # delta = none
                        # default='/scratch/users/stephen.lu/projects/protevo/logs/train/runs/2026-01-06_18-35-44/checkpoints/epoch_001.ckpt',    # delta = 0.5
                        help='Path to CTMC model checkpoint')

    # Oracle seed selection
    parser.add_argument('--oracle', type=str, default='SARSCoV2Beta',
                        choices=['SARSCoV1', 'SARSCoV2Beta'],
                        help='Oracle variant for seed selection (default: SARSCoV2Beta)')
    parser.add_argument('--oracle-seed-idx', type=int, default=0,
                        help='Index of oracle seed sequence to use (must be binder, default: 0)')

    # Sampling parameters
    parser.add_argument('--branch-length', type=float, required=True,
                        help='Fixed evolutionary time for all trajectories')
    parser.add_argument('--use-discrete-steps', action='store_true',
                        help='Use discrete number of steps in Gillespie sampling (instead of continuous time)')
    parser.add_argument('--max-mutations', type=int, default=None,
                        help='Maximum number of mutations allowed (optional, default: None for unlimited)')
    parser.add_argument('--batch-size', type=int, default=100,
                        help='Number of independent trajectories to sample (default: 100)')
    parser.add_argument('--mask-region', type=str, default=None,
                        choices=['CDR1', 'CDR2', 'CDR3', 'CDR_overall', 'FR1', 'FR2', 'FR3', 'FR4', 'FR_overall', None],
                        help='Restrict mutations to specific antibody region (default: None, allows all positions)')

    # Guided sampling
    parser.add_argument('--use-guided', action='store_true',
                        help='Enable TAG guided sampling (in addition to unguided)')
    parser.add_argument('--guidance-strength', type=float, default=2.0,
                        help='Guidance strength γ for TAG (default: 2.0)')
    parser.add_argument('--oracle-chunk-size', type=int, default=5000,
                        help='Max sequences per oracle call (default: 5000)')

    # Internal sampling parameters
    parser.add_argument('--max-decode-steps', type=int, default=2048,
                        help='Maximum Gillespie steps per trajectory (default: 2048)')
    parser.add_argument('--n-sequences', type=int, default=1,
                        help='Sequences per sampling attempt for length rejection (default: 5)')
    parser.add_argument('--max-retries', type=int, default=100,
                        help='Max retries for mutation ceiling rejection sampling (default: 100)')

    # Output
    parser.add_argument('--save-csv', action='store_true', help='Whether to save output as CSV')
    parser.add_argument('--output-path', type=str, default=None,
                        help='Output CSV path (default: auto-generated in current dir)')
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
    module_dir = Path("/accounts/projects/yss/stephen.lu/peint-workspace/main/evo/evo/oracles/data")
    csv_path = module_dir / csv_files[oracle_variant]

    if not csv_path.exists():
        raise FileNotFoundError(f"Oracle CSV not found: {csv_path}")

    # Load CSV
    df = pd.read_csv(csv_path)
    df.columns = ['sequence', 'binding']

    # Filter to binders only
    binders = df[df['binding'] == 1].reset_index(drop=True)

    print(f"Loaded {len(df)} total sequences from {csv_files[oracle_variant]}")
    print(f"Filtered to {len(binders)} binders (binding=1)")

    return binders


def select_oracle_seed(oracle_variant: str, seed_idx: int) -> str:
    """Select an oracle seed sequence (no germline mapping).

    Parameters
    ----------
    oracle_variant : str
        Oracle variant name
    seed_idx : int
        Index of seed to use

    Returns
    -------
    str
        Seed sequence
    """
    binders = load_oracle_binders(oracle_variant)

    if seed_idx < 0 or seed_idx >= len(binders):
        raise ValueError(f"seed_idx {seed_idx} out of range [0, {len(binders)-1}]")

    seed_sequence = binders.iloc[seed_idx]['sequence']

    print(f"\n{'='*80}")
    print("ORACLE SEED SELECTION")
    print(f"{'='*80}")
    print(f"Oracle: {oracle_variant}")
    print(f"Seed index: {seed_idx}")
    print(f"Seed sequence ({len(seed_sequence)} aa):")
    print(f"  {seed_sequence}")
    print(f"{'='*80}\n")

    return seed_sequence


def generate_random_protein_sequence(length: int, seed: int = None) -> str:
    """Generate a random protein sequence of specified length.

    Parameters
    ----------
    length : int
        Desired sequence length
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    str
        Random protein sequence using standard amino acids
    """
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'  # Standard 20 amino acids
    if seed is not None:
        rng = random.Random(seed)
        return ''.join(rng.choice(amino_acids) for _ in range(length))
    else:
        return ''.join(random.choice(amino_acids) for _ in range(length))


def count_mutations(seq1: str, seq2: str) -> int:
    """Count number of mutations between two sequences."""
    return sum(1 for a, b in zip(seq1, seq2) if a != b)


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


def decode_sequence_from_toks(toks, vocab, skip_gap_tokens=False):
    tokens = []
    gap_idx = vocab.tokens_to_idx.get("-", -1)
    for tok in toks:
        if tok == vocab.bos_idx:
            continue
        if skip_gap_tokens and tok == gap_idx:
            continue
        if tok == vocab.eos_idx or tok == vocab.pad_idx:
            break
        tokens.append(vocab.token(tok))
    return "".join(tokens)  


def calculate_hamming_distance(seq1: str, seq2: str) -> int:
    """Calculate Hamming distance between two sequences of equal length."""
    if len(seq1) != len(seq2):
        raise ValueError("Sequences must be of equal length to compute Hamming distance.")
    return sum(c1 != c2 for c1, c2 in zip(seq1, seq2))  


def sample_single_trajectory(
    generate_fn,
    seed_seq: str,
    branch_length: float,
    vocab: Vocab,
    device,
    n_sequences: int,
    max_decode_steps: int,
    max_retries: int = 100,
    mask: Optional[Tensor] = None,
    return_sampling_time: bool = False
) -> tuple[str, int] | tuple[str, int, float]:
    """Sample a single trajectory using rejection sampling for length filtering.

    Parameters
    ----------
    generate_fn : callable
        Generation function (unguided or guided)
    seed_seq : str
        Starting sequence
    branch_length : float
        Evolutionary time
    vocab : Vocab
        Vocabulary for encoding/decoding
    device : torch.device
        Device for computation
    n_sequences : int
        Number of candidates per attempt
    max_decode_steps : int
        Max Gillespie steps
    max_retries : int
        Max rejection sampling attempts
    mask : Tensor, optional
        Boolean mask for region-specific mutations
    return_sampling_time : bool
        If True, return sampling time as 3rd element

    Returns
    -------
    tuple
        (sampled_sequence, retry_count) or
        (sampled_sequence, retry_count, sampling_time) if return_sampling_time=True
    """
    root_length = len(seed_seq)
    total_sampling_time = 0.0

    for retry in range(max_retries):
        # Encode seed sequence
        parent_encoded = vocab.encode_single_sequence(seed_seq)
        x_rep = torch.from_numpy(parent_encoded).unsqueeze(0).repeat(n_sequences, 1).to(device)
        t_rep = torch.tensor([branch_length] * n_sequences, device=device).repeat(n_sequences, 1).to(device)
        t_rep = t_rep.view(-1, 1)
        x_sizes = torch.tensor([len(seed_seq) + 1], dtype=torch.long)
        x_sizes[0] += vocab.prepend_bos
        x_sizes[-1] += vocab.append_eos - 1
        x_sizes = torch.nn.functional.pad(x_sizes, (0, len(x_rep[0]) - len(x_sizes)), value=0).view(1, -1)
        x_sizes = x_sizes.repeat(n_sequences, 1).to(device)

        # Expand mask if provided
        mask_rep = None
        if mask is not None:
            mask_rep = mask.repeat(n_sequences, 1)

        # Generate candidates
        sampling_start = time.perf_counter()
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type=="cuda"):
            y_rep = generate_fn(
                x=x_rep,
                t=t_rep,
                device=device,
                p=None,
                max_decode_steps=max_decode_steps,
                x_sizes=x_sizes,
                mask=mask_rep
            )
        sampling_end = time.perf_counter()
        total_sampling_time += (sampling_end - sampling_start)

        # Decode candidates
        candidates = []
        for i in range(n_sequences):
            seq_str = decode_sequence_from_toks(y_rep[i].cpu().numpy(), vocab)
            candidates.append(seq_str)

        # Filter by length
        valid_candidates = [s for s in candidates if len(s) == root_length]

        if len(valid_candidates) > 0:
            if return_sampling_time:
                return random.choice(valid_candidates), retry, total_sampling_time
            else:
                return random.choice(valid_candidates), retry

    # Fallback: return seed if all retries fail
    print(f"WARNING: All {max_retries} attempts failed length filter, returning seed sequence")
    if return_sampling_time:
        return seed_seq, max_retries, total_sampling_time
    else:
        return seed_seq, max_retries


def sample_with_mutation_ceiling(
    generate_fn,
    seed_seq: str,
    branch_length: float,
    vocab: Vocab,
    device,
    n_sequences: int,
    max_decode_steps: int,
    max_mutations: Optional[int] = None,
    max_retries: int = 100,
    mask: Optional[Tensor] = None,
    trajectory_idx: int = 0,
    return_sampling_time: bool = False
) -> tuple[str, int, int] | tuple[str, int, int, float]:
    """Sample trajectory with optional mutation ceiling using post-hoc rejection.

    Parameters
    ----------
    generate_fn : callable
        Generation function
    seed_seq : str
        Starting sequence
    branch_length : float
        Evolutionary time
    vocab : Vocab
        Vocabulary
    device : torch.device
        Device
    n_sequences : int
        Candidates per attempt
    max_decode_steps : int
        Max Gillespie steps
    max_mutations : int, optional
        Mutation ceiling (None for unlimited)
    max_retries : int
        Max rejection attempts
    mask : Tensor, optional
        Boolean mask for region-specific mutations
    trajectory_idx : int
        Trajectory index for error messages
    return_sampling_time : bool
        If True, return sampling time as 4th element

    Returns
    -------
    tuple
        (sampled_sequence, n_mutations, total_retries) or
        (sampled_sequence, n_mutations, total_retries, sampling_time) if return_sampling_time=True
    """
    total_retries = 0
    total_sampling_time = 0.0

    for ceiling_retry in range(max_retries):
        # Sample a trajectory
        if return_sampling_time:
            sampled_seq, length_retries, sampling_time = sample_single_trajectory(
                generate_fn=generate_fn,
                seed_seq=seed_seq,
                branch_length=branch_length,
                vocab=vocab,
                device=device,
                n_sequences=n_sequences,
                max_decode_steps=max_decode_steps,
                max_retries=max_retries,
                mask=mask,
                return_sampling_time=True
            )
            total_sampling_time += sampling_time
        else:
            sampled_seq, length_retries = sample_single_trajectory(
                generate_fn=generate_fn,
                seed_seq=seed_seq,
                branch_length=branch_length,
                vocab=vocab,
                device=device,
                n_sequences=n_sequences,
                max_decode_steps=max_decode_steps,
                max_retries=max_retries,
                mask=mask
            )

        total_retries += length_retries + 1

        # Count mutations
        n_mutations = count_mutations(sampled_seq, seed_seq)

        # Check mutation ceiling
        if max_mutations is None or n_mutations <= max_mutations:
            if return_sampling_time:
                return sampled_seq, n_mutations, total_retries, total_sampling_time
            else:
                return sampled_seq, n_mutations, total_retries

    # Fail-safe: if we've exhausted all retries, raise an error
    error_msg = (
        f"\n{'='*80}\n"
        f"FATAL ERROR: Mutation ceiling cannot be satisfied\n"
        f"{'='*80}\n"
        f"Trajectory index: {trajectory_idx}\n"
        f"Branch length: {branch_length}\n"
        f"Max mutations allowed: {max_mutations}\n"
        f"Max retries attempted: {max_retries}\n"
        f"\n"
        f"After {max_retries} attempts, no sampled sequence had ≤ {max_mutations} mutations.\n"
        f"This suggests the branch length ({branch_length}) is too large for the mutation\n"
        f"ceiling ({max_mutations}). Please either:\n"
        f"  1. Increase --max-mutations to a higher value\n"
        f"  2. Decrease --branch-length to a smaller value\n"
        f"  3. Increase --max-retries (currently {max_retries})\n"
        f"\n"
        f"The script is exiting to prevent infinite resampling.\n"
        f"{'='*80}\n"
    )
    print(error_msg)
    raise RuntimeError(f"Mutation ceiling of {max_mutations} not achievable after {max_retries} attempts")


def main():
    args = parse_args()

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Set random seed
    random_seed = args.seed if args.seed is not None else int(time.time())
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    # Load model
    print(f"Loading CTMC model from: {args.model_path}")
    module = CTMCModule.load_from_checkpoint(str(args.model_path), map_location=device, strict=False)
    net: NeuralCTMC = module.net
    vocab: Vocab = net.vocab
    net = net.eval().to(device)
    generator = NeuralCTMCGenerator(neural_ctmc=net)
    print("Model loaded successfully\n")

    # Select oracle seed (no germline mapping)
    seed_seq = select_oracle_seed(
        oracle_variant=args.oracle,
        seed_idx=args.oracle_seed_idx
    )

    # Create region mask if requested
    mask_tensor = None
    if args.mask_region is not None:
        print(f"Creating region mask for: {args.mask_region}")
        region_masks_raw = create_region_masks(seed_seq, scheme="imgt")

        # Pad mask to include special tokens (BOS at start, EOS at end)
        # Special token positions should be False (not mutable)
        mask_array = region_masks_raw[args.mask_region]
        padded_mask = np.pad(mask_array, (int(vocab.prepend_bos), int(vocab.append_eos)), constant_values=False)
        mask_tensor = torch.from_numpy(padded_mask).unsqueeze(0).to(device)

        # Display mask information
        n_positions = mask_array.sum()
        pct = (n_positions / len(mask_array)) * 100
        print(f"  {args.mask_region}: {n_positions} positions ({pct:.1f}% of sequence)")
        print(f"  Mutations will be restricted to this region\n")

    # Load oracle if using guided sampling
    print(f"Loading oracle ({args.oracle}) for guided sampling...")
    oracle = OracleCallCounter(get_oracle(
        args.oracle,
        enable_mc_dropout=True,
        mc_samples=10,
        mc_dropout_seed=42,
        use_iglm_weighting=False,
        device=device,
        precompute_seed_fitnesses=False,
    ))
    print("Oracle loaded successfully\n")

    seed_fitness = oracle.predict(seed_seq, increment=False)[0]
    print(f"Seed Fitness: {seed_fitness}")

    # Print experiment summary
    print(f"{'='*80}")
    print("SAMPLING EXPERIMENT")
    print(f"{'='*80}")
    print(f"Seed sequence length: {len(seed_seq)} aa")

    if args.use_discrete_steps:
        print(f"Using {args.max_mutations} number of discrete steps in Gillespie sampling")
        assert args.max_mutations is not None, "When using discrete steps, --max-mutations must be specified"
        args.branch_length = int(args.max_mutations)
    else:
        print(f"Using continuous time in Gillespie sampling")
        print(f"Branch length: {args.branch_length}")

    print(f"Max mutations: {args.max_mutations if args.max_mutations else 'unlimited'}")
    print(f"Region mask: {args.mask_region if args.mask_region else 'None (all positions allowed)'}")
    print(f"Batch size: {args.batch_size} trajectories")
    print(f"Methods: unguided" + (" + guided (TAG)" if args.use_guided else ""))
    if args.use_guided:
        print(f"Guidance strength: {args.guidance_strength}")
    print(f"Random seed: {random_seed}")
    print(f"{'='*80}\n")

    # Define generation functions
    def unguided_generate_fn(x: Tensor, t: Tensor, device, p, max_decode_steps, x_sizes, mask=None, **kwargs):
        return generator.generate_with_gillespie(
            t=t, x=x, x_sizes=x_sizes,
            max_decode_steps=max_decode_steps,
            use_scalar_steps=args.use_discrete_steps,
            mask=mask,
        )

    def guided_generate_fn(x: Tensor, t: Tensor, device, p, max_decode_steps, x_sizes, mask=None, **kwargs):
        return generator.generate_with_gillespie(
            t=t, x=x, x_sizes=x_sizes,
            oracle=oracle,
            guidance_strength=args.guidance_strength,
            use_taylor_approx=True,
            max_decode_steps=max_decode_steps,
            use_scalar_steps=args.use_discrete_steps,
            oracle_chunk_size=args.oracle_chunk_size,
            use_guidance=True,
            mask=mask,
        )

    # Sample trajectories
    results = []
    methods = []
    if args.use_guided:
        methods.append('guided')
    else:
        methods.append('unguided')

    for method in methods:
        unique_seq_set = set()
        print(f"\nSampling {args.batch_size} trajectories using {method} method...")
        generate_fn = unguided_generate_fn if method == 'unguided' else guided_generate_fn

        for traj_idx in tqdm(range(args.batch_size), desc=f"{method.capitalize()}"):
            start_time = time.time()
            # set a different seed for each trajectory for diversity
            random.seed(random_seed + traj_idx)
            np.random.seed(random_seed + traj_idx)
            torch.manual_seed(random_seed + traj_idx)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(random_seed + traj_idx)
            sampled_seq, n_mutations, total_retries = sample_with_mutation_ceiling(
                generate_fn=generate_fn,
                seed_seq=seed_seq,
                branch_length=args.branch_length,
                vocab=vocab,
                device=device,
                n_sequences=args.n_sequences,
                max_decode_steps=args.max_decode_steps,
                max_mutations=args.max_mutations,
                max_retries=args.max_retries,
                mask=mask_tensor,
                trajectory_idx=traj_idx,
            )
            end_time = time.time()
            time_taken = end_time - start_time
            sampled_seq_fitness = oracle.predict(sampled_seq, increment=False)[0]
            results.append({
                'seed_seq': seed_seq,
                'branch_length': args.branch_length,
                'guidance_type': method,
                'sampled_seq': sampled_seq,
                'n_mutations': n_mutations,
                'total_retries': total_retries,
                'trajectory_idx': traj_idx,
                'time': time_taken,
                'final_fitness': sampled_seq_fitness,
                'delta_fitness': sampled_seq_fitness - seed_fitness,
            })
            if traj_idx % 10 == 0:
                print(f"Number of unique sequences so far: {len(unique_seq_set)} / {traj_idx}")

            unique_seq_set.add(sampled_seq)

    # Create results dataframe
    results_df = pd.DataFrame(results)

    # Calculate additional sequence level properties like oasis humanness and edit distance from the seed
    all_sequences = results_df['sampled_seq'].tolist()
    oasis_humanness = compute_oasis_humanness(all_sequences)
    iglm_humanness = compute_iglm_humanness(all_sequences)
    edit_dist_from_seed = [calculate_hamming_distance(seq, seed_seq) for seq in all_sequences]
    results_df['oasis_humanness'] = oasis_humanness
    results_df['iglm_humanness'] = iglm_humanness
    results_df['edit_dist_from_seed'] = edit_dist_from_seed

    # Generate output path if not provided
    if args.output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        oracle_name = args.oracle.replace('SARSCoV', 'SC')
        ceiling_str = f"_maxmut{args.max_mutations}" if args.max_mutations else ""
        mask_str = f"_mask{args.mask_region}" if args.mask_region else ""
        output_path = f"cosine_{oracle_name}_seed{args.oracle_seed_idx}_t{args.branch_length}{ceiling_str}{mask_str}_{timestamp}.csv"
    else:
        output_path = args.output_path

    # Save results
    if args.save_csv:
        results_df.to_csv(output_path, index=False)

    # Print summary statistics
    print("\nSummary statistics:")
    for method in methods:
        method_seqs = results_df[results_df['guidance_type'] == method]['sampled_seq'].tolist()
        method_df = results_df[results_df['guidance_type'] == method]
        n_seqs = len(method_df)
        avg_fitness = method_df['final_fitness'].mean()
        max_fitness = method_df['final_fitness'].max()
        unique_seqs = method_df['sampled_seq'].nunique()
        total_time = method_df['time'].sum()
        time_per_sample = method_df['time'].mean()
        int_div_mean, int_div_std = calculate_internal_diversity(method_seqs)

        print(f"\n{'='*60}")
        print(f"Method: {method.capitalize()}")
        print(f"{'='*60}")
        if args.save_csv:
            print(f"Results saved to: {output_path}")
        print(f"Final Population Size: {len(method_df)}")
        print(f"Unique Sequences: {unique_seqs} / {n_seqs}")
        print(f"Proportion Unique: {unique_seqs / n_seqs:.4f}")
        print(f"Avg Fitness: {avg_fitness:.4f} +/- {method_df['final_fitness'].std():.4f}")
        print(f"Max Fitness: {max_fitness:.4f}")
        print(f"Avg OASIS Humanness: {method_df['oasis_humanness'].mean():.4f} +/- {method_df['oasis_humanness'].std():.4f}")
        print(f"Avg IGLM Humanness: {method_df['iglm_humanness'].mean():.4f} +/- {method_df['iglm_humanness'].std():.4f}")
        print(f"Delta Fitness (Avg): {avg_fitness - seed_fitness:.4f}")
        print(f"Delta Fitness (Max): {max_fitness - seed_fitness:.4f}")
        print(f"Internal Diversity (Avg Hamming Dist): {int_div_mean:.4f} +/- {int_div_std:.4f}")
        print(f"Avg Mutations: {method_df['n_mutations'].mean():.4f} +/- {method_df['n_mutations'].std():.4f}")
        print(f"Total Runtime: {total_time:.4f}s")
        print(f"Avg Time per Sample: {time_per_sample:.4f}s")
        print(f"Total oracle calls: {oracle.get_call_count()}")
        print(f"Avg oracle calls per sample: {oracle.get_call_count() / n_seqs:.4f}")


if __name__ == "__main__":
    main()
