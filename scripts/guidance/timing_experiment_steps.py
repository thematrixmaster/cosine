"""Timing experiment sweeping over number of Gillespie steps.

This script measures the runtime of guided vs unguided sampling across different
step counts: [5, 10, 25, 50, 100].

Usage:
    uv run python scripts/guidance/timing_experiment_steps.py \
        --oracle SARSCoV2Beta \
        --oracle-seed-idx 0 \
        --seed 42
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

import numpy as np
import torch

from evo.tokenization import Vocab
from evo.oracles import get_oracle
from peint.models.modules.ctmc_module import CTMCModule
from peint.models.nets.ctmc import NeuralCTMC, NeuralCTMCGenerator
from oracle_counter import OracleCallCounter

# Import functions from cosine.py
from cosine import (
    select_oracle_seed,
    sample_with_mutation_ceiling,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Timing experiment: sweep over number of Gillespie steps'
    )

    # Model
    parser.add_argument('--model-path', type=str,
                        default='/scratch/users/stephen.lu/projects/protevo/logs/train/runs/2026-01-06_18-32-49/checkpoints/epoch_001.ckpt',
                        help='Path to CTMC model checkpoint')

    # Oracle seed selection
    parser.add_argument('--oracle', type=str, default='SARSCoV2Beta',
                        choices=['SARSCoV1', 'SARSCoV2Beta'],
                        help='Oracle variant for seed selection')
    parser.add_argument('--oracle-seed-idx', type=int, default=0,
                        help='Index of oracle seed sequence to use')

    # Sampling parameters
    parser.add_argument('--guidance-strength', type=float, default=2.0,
                        help='Guidance strength γ for TAG (default: 2.0)')
    parser.add_argument('--oracle-chunk-size', type=int, default=5000,
                        help='Max sequences per oracle call (default: 5000)')
    parser.add_argument('--max-decode-steps', type=int, default=2048,
                        help='Maximum Gillespie steps per trajectory (default: 2048)')
    parser.add_argument('--n-sequences', type=int, default=1,
                        help='Sequences per sampling attempt for length rejection (default: 1)')
    parser.add_argument('--max-retries', type=int, default=100,
                        help='Max retries for mutation ceiling rejection sampling (default: 100)')

    # Experiment parameters
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility (default: None, uses timestamp)')

    return parser.parse_args()


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

    # Select oracle seed
    seed_seq = select_oracle_seed(
        oracle_variant=args.oracle,
        seed_idx=args.oracle_seed_idx
    )

    # Load oracle
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

    # Define generation functions
    def unguided_generate_fn(x, t, device, p, max_decode_steps, x_sizes, mask=None, **kwargs):
        return generator.generate_with_gillespie(
            t=t, x=x, x_sizes=x_sizes,
            max_decode_steps=max_decode_steps,
            use_scalar_steps=True,  # Use discrete steps
            mask=mask,
        )

    def guided_generate_fn(x, t, device, p, max_decode_steps, x_sizes, mask=None, **kwargs):
        return generator.generate_with_gillespie(
            t=t, x=x, x_sizes=x_sizes,
            oracle=oracle.oracle,  # Pass underlying oracle
            guidance_strength=args.guidance_strength,
            use_taylor_approx=True,
            max_decode_steps=max_decode_steps,
            use_scalar_steps=True,  # Use discrete steps
            oracle_chunk_size=args.oracle_chunk_size,
            use_guidance=True,
            mask=mask,
        )

    # Experiment configuration
    step_counts = [5, 10, 25, 50, 100]
    timing_results = []

    print("=" * 80)
    print("RUNNING TIMING EXPERIMENT - STEP COUNT SWEEP")
    print("=" * 80)
    print(f"Seed sequence: {seed_seq[:50]}... (length={len(seed_seq)})")
    print(f"Step counts: {step_counts}")
    print(f"Random seed: {random_seed}\n")

    # Warmup run
    print("Running warmup (5 steps guided)...")
    warmup_start = time.perf_counter()
    _ = sample_with_mutation_ceiling(
        generate_fn=guided_generate_fn,
        seed_seq=seed_seq,
        branch_length=5.0,
        vocab=vocab,
        device=device,
        n_sequences=args.n_sequences,
        max_decode_steps=args.max_decode_steps,
        max_mutations=None,
        max_retries=args.max_retries,
        mask=None,
        trajectory_idx=0,
        return_sampling_time=False
    )
    warmup_time = time.perf_counter() - warmup_start
    print(f"Warmup completed in {warmup_time:.4f}s\n")

    # Run experiments for both unguided and guided
    for method_name, generate_fn in [('Unguided', unguided_generate_fn), ('Guided', guided_generate_fn)]:
        print(f"\n{'='*80}")
        print(f"{method_name} Sampling")
        print(f"{'='*80}")

        for step_count in step_counts:
            print(f"\n  Step Count: {step_count}")

            # Set random seed for reproducibility
            random.seed(random_seed)
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(random_seed)

            # Sample with timing
            sampled_seq, n_mutations, total_retries, sampling_time = sample_with_mutation_ceiling(
                generate_fn=generate_fn,
                seed_seq=seed_seq,
                branch_length=float(step_count),
                vocab=vocab,
                device=device,
                n_sequences=args.n_sequences,
                max_decode_steps=args.max_decode_steps,
                max_mutations=None,
                max_retries=args.max_retries,
                mask=None,
                trajectory_idx=0,
                return_sampling_time=True
            )

            # Evaluate fitness
            sampled_seq_fitness = oracle.predict(sampled_seq, increment=False)[0]

            timing_results.append({
                'method': method_name,
                'step_count': step_count,
                'sampling_time': sampling_time,
                'final_fitness': sampled_seq_fitness,
                'n_mutations': n_mutations,
            })

            print(f"  Sampling Time: {sampling_time:.4f}s | Mutations: {n_mutations} | Fitness: {sampled_seq_fitness:.4f}")

    # Print final results table
    print("\n" + "=" * 80)
    print("TIMING EXPERIMENT RESULTS (Guided vs Unguided)")
    print("=" * 80)

    # Organize results by step count
    results_by_step = {}
    for result in timing_results:
        step = result['step_count']
        if step not in results_by_step:
            results_by_step[step] = {}
        results_by_step[step][result['method']] = result

    # Print header
    print(f"{'Steps':<8} | {'Unguided (s)':<14} {'Guided (s)':<14} {'Speedup':<10} | {'Fitness (U)':<12} {'Fitness (G)':<12} | {'Mutations':<10}")
    print("-" * 110)

    # Print rows
    for step in sorted(results_by_step.keys()):
        unguided = results_by_step[step].get('Unguided', {})
        guided = results_by_step[step].get('Guided', {})

        u_time = unguided.get('sampling_time', 0)
        g_time = guided.get('sampling_time', 0)
        speedup = u_time / g_time if g_time > 0 else 0

        u_fitness = unguided.get('final_fitness', 0)
        g_fitness = guided.get('final_fitness', 0)

        mutations = guided.get('n_mutations', unguided.get('n_mutations', 0))

        print(f"{step:<8} | {u_time:<14.4f} {g_time:<14.4f} {speedup:<10.2f} | {u_fitness:<12.4f} {g_fitness:<12.4f} | {mutations:<10}")

    print("=" * 110)


if __name__ == "__main__":
    main()
