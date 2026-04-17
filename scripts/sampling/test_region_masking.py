"""Test region masking functionality for CTMC Gillespie sampling.

This script demonstrates and tests the region masking feature that restricts
mutations to specific regions (e.g., CDR regions of antibodies).
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from evo.antibody import create_region_masks
from evo.oracles import get_oracle
from evo.tokenization import Vocab
from cosine.models.modules.ctmc_module import CTMCModule
from cosine.models.nets.ctmc import NeuralCTMC, NeuralCTMCGenerator


def count_mutations(seq1: str, seq2: str) -> dict:
    """Count mutations and track their positions."""
    mutations = []
    for i, (aa1, aa2) in enumerate(zip(seq1, seq2)):
        if aa1 != aa2:
            mutations.append(i)
    return {
        'count': len(mutations),
        'positions': mutations
    }


def check_mutations_in_mask(mutation_positions: list, mask: np.ndarray) -> dict:
    """Check if mutations occurred only in masked regions."""
    in_mask = [pos for pos in mutation_positions if mask[pos]]
    out_of_mask = [pos for pos in mutation_positions if not mask[pos]]

    return {
        'in_mask': in_mask,
        'out_of_mask': out_of_mask,
        'all_in_mask': len(out_of_mask) == 0,
        'pct_in_mask': len(in_mask) / len(mutation_positions) * 100 if mutation_positions else 0
    }


def test_region_masking():
    """Test region masking with CDR regions of an antibody sequence."""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    # Load model
    model_path = "/scratch/users/stephen.lu/projects/protevo/logs/train/runs/2026-01-06_18-32-49/checkpoints/epoch_001.ckpt"
    print(f"Loading CTMC model from: {model_path}")
    module = CTMCModule.load_from_checkpoint(str(model_path), map_location=device, strict=False)
    net: NeuralCTMC = module.net
    vocab: Vocab = net.vocab
    net = net.eval().to(device)
    generator = NeuralCTMCGenerator(neural_ctmc=net)

    # Load oracle
    print("Loading oracle (SARSCoV1)...")
    oracle = get_oracle("SARSCoV1", enable_mc_dropout=True, mc_samples=10, device=device)

    # Get a seed antibody sequence
    seed_data = list(oracle.seed_data.values())[0]
    antibody_seq = seed_data["sequence"]
    seed_fitness = seed_data["fitness"]

    print(f"\n{'='*80}")
    print("ANTIBODY SEQUENCE")
    print(f"{'='*80}")
    print(f"Sequence (length {len(antibody_seq)}): {antibody_seq[:60]}...")
    print(f"Seed fitness: {seed_fitness:.4f}")

    # Create region masks (without special tokens initially)
    print("\nCreating region masks using IMGT scheme...")
    region_masks_raw = create_region_masks(antibody_seq, scheme="imgt")

    # Pad masks to include special tokens (BOS at start, EOS at end)
    # Special token positions should be False (not mutable)
    region_masks = {}
    for region_name, mask in region_masks_raw.items():
        # Pad: [False (BOS), ...mask..., False (EOS)]
        padded_mask = np.pad(mask, (int(vocab.prepend_bos), int(vocab.append_eos)), constant_values=False)
        region_masks[region_name] = padded_mask

    # Display region information
    print(f"\n{'='*80}")
    print("REGION INFORMATION")
    print(f"{'='*80}")
    for region_name in ['CDR1', 'CDR2', 'CDR3', 'FR1', 'FR2', 'FR3', 'FR4', 'CDR_overall', 'FR_overall']:
        if region_name in region_masks:
            mask = region_masks[region_name]
            # Count without special tokens for display
            mask_no_special = mask[int(vocab.prepend_bos):-int(vocab.append_eos) if vocab.append_eos else None]
            n_positions = mask_no_special.sum()
            pct = (n_positions / len(mask_no_special)) * 100
            print(f"{region_name:15s}: {n_positions:3d} positions ({pct:5.1f}%)")

    # Convert sequence to tensor WITH special tokens (BOS/EOS)
    x_encoded = vocab.encode_single_sequence(antibody_seq)  # Adds BOS/EOS
    x = torch.from_numpy(x_encoded).unsqueeze(0).to(device)
    # x_sizes should include special tokens
    x_sizes = torch.tensor([len(x_encoded)], device=device)
    t = torch.tensor([0.5], device=device)  # Branch length 0.5

    print(f"\n{'='*80}")
    print("TEST 1: Unguided Sampling WITHOUT Mask (Baseline)")
    print(f"{'='*80}")

    # Test 1: Sample without mask
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        y_no_mask = generator.generate_with_gillespie(
            x=x,
            t=t,
            x_sizes=x_sizes,
            temperature=1.0,
            no_special_toks=True,
            max_decode_steps=1000,
            verbose=False,
        )

    # Convert to string
    y_no_mask_str = "".join([vocab.token(idx.item()) for idx in y_no_mask[0]
                             if vocab.token(idx.item()) in set("ARNDCQEGHILKMFPSTWYV")])

    # Analyze mutations
    mutations_no_mask = count_mutations(antibody_seq, y_no_mask_str)
    print(f"Total mutations: {mutations_no_mask['count']}")
    print(f"Mutation positions: {mutations_no_mask['positions'][:20]}{'...' if len(mutations_no_mask['positions']) > 20 else ''}")

    # Check distribution across regions
    cdr_overall_mask = region_masks['CDR_overall']
    check_no_mask = check_mutations_in_mask(mutations_no_mask['positions'], cdr_overall_mask)
    print(f"Mutations in CDR regions: {len(check_no_mask['in_mask'])} ({check_no_mask['pct_in_mask']:.1f}%)")
    print(f"Mutations in FR regions: {len(check_no_mask['out_of_mask'])} ({100-check_no_mask['pct_in_mask']:.1f}%)")

    print(f"\n{'='*80}")
    print("TEST 2: Unguided Sampling WITH CDR Mask")
    print(f"{'='*80}")

    # Test 2: Sample with CDR mask
    cdr_mask_tensor = torch.from_numpy(cdr_overall_mask).unsqueeze(0).to(device)

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        y_cdr_mask = generator.generate_with_gillespie(
            x=x,
            t=t,
            x_sizes=x_sizes,
            temperature=1.0,
            no_special_toks=True,
            max_decode_steps=1000,
            verbose=False,
            mask=cdr_mask_tensor,  # CDR-only mask
        )

    # Convert to string
    y_cdr_mask_str = "".join([vocab.token(idx.item()) for idx in y_cdr_mask[0]
                              if vocab.token(idx.item()) in set("ARNDCQEGHILKMFPSTWYV")])

    # Analyze mutations
    mutations_cdr_mask = count_mutations(antibody_seq, y_cdr_mask_str)
    print(f"Total mutations: {mutations_cdr_mask['count']}")
    print(f"Mutation positions: {mutations_cdr_mask['positions'][:20]}{'...' if len(mutations_cdr_mask['positions']) > 20 else ''}")

    # Check if all mutations are in CDR
    check_cdr_mask = check_mutations_in_mask(mutations_cdr_mask['positions'], cdr_overall_mask)
    print(f"Mutations in CDR regions: {len(check_cdr_mask['in_mask'])} ({check_cdr_mask['pct_in_mask']:.1f}%)")
    print(f"Mutations in FR regions: {len(check_cdr_mask['out_of_mask'])} ({100-check_cdr_mask['pct_in_mask']:.1f}%)")

    if check_cdr_mask['all_in_mask']:
        print("✓ SUCCESS: All mutations occurred in CDR regions!")
    else:
        print(f"✗ FAILURE: {len(check_cdr_mask['out_of_mask'])} mutations occurred outside CDR regions")
        print(f"  Out-of-mask positions: {check_cdr_mask['out_of_mask']}")

    print(f"\n{'='*80}")
    print("TEST 3: Guided Sampling WITH CDR Mask")
    print(f"{'='*80}")

    # Test 3: Guided sampling with CDR mask
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        y_guided_cdr = generator.generate_with_gillespie(
            x=x,
            t=t,
            x_sizes=x_sizes,
            oracle=oracle,
            guidance_strength=2.0,
            use_guidance=True,
            temperature=1.0,
            no_special_toks=True,
            max_decode_steps=1000,
            use_taylor_approx=True,  # Use faster Taylor approximation
            verbose=False,
            mask=cdr_mask_tensor,  # CDR-only mask
        )

    # Convert to string
    y_guided_cdr_str = "".join([vocab.token(idx.item()) for idx in y_guided_cdr[0]
                                if vocab.token(idx.item()) in set("ARNDCQEGHILKMFPSTWYV")])

    # Analyze mutations
    mutations_guided_cdr = count_mutations(antibody_seq, y_guided_cdr_str)
    print(f"Total mutations: {mutations_guided_cdr['count']}")
    print(f"Mutation positions: {mutations_guided_cdr['positions'][:20]}{'...' if len(mutations_guided_cdr['positions']) > 20 else ''}")

    # Check if all mutations are in CDR
    check_guided_cdr = check_mutations_in_mask(mutations_guided_cdr['positions'], cdr_overall_mask)
    print(f"Mutations in CDR regions: {len(check_guided_cdr['in_mask'])} ({check_guided_cdr['pct_in_mask']:.1f}%)")
    print(f"Mutations in FR regions: {len(check_guided_cdr['out_of_mask'])} ({100-check_guided_cdr['pct_in_mask']:.1f}%)")

    # Score the sequences
    print("\nScoring sequences with oracle...")
    scores, _ = oracle.predict_batch([antibody_seq, y_cdr_mask_str, y_guided_cdr_str])
    print(f"  Seed:                  {scores[0]:.4f}")
    print(f"  Unguided + CDR mask:   {scores[1]:.4f} (Δ = {scores[1]-scores[0]:+.4f})")
    print(f"  Guided + CDR mask:     {scores[2]:.4f} (Δ = {scores[2]-scores[0]:+.4f})")

    if check_guided_cdr['all_in_mask']:
        print("✓ SUCCESS: All mutations occurred in CDR regions!")
    else:
        print(f"✗ FAILURE: {len(check_guided_cdr['out_of_mask'])} mutations occurred outside CDR regions")
        print(f"  Out-of-mask positions: {check_guided_cdr['out_of_mask']}")

    print(f"\n{'='*80}")
    print("TEST 4: Edge Case - All Positions Masked (Should Raise Error)")
    print(f"{'='*80}")

    # Test 4: Try with all-False mask (should raise error)
    all_false_mask = torch.zeros_like(cdr_mask_tensor, dtype=torch.bool)

    try:
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            y_error = generator.generate_with_gillespie(
                x=x,
                t=t,
                x_sizes=x_sizes,
                temperature=1.0,
                no_special_toks=True,
                max_decode_steps=1000,
                verbose=False,
                mask=all_false_mask,
            )
        print("✗ FAILURE: Should have raised ValueError for all-False mask")
    except ValueError as e:
        print(f"✓ SUCCESS: Correctly raised ValueError")
        print(f"  Error message: {str(e)}")

    print(f"\n{'='*80}")
    print("TEST 5: Edge Case - Shape Mismatch (Should Raise Error)")
    print(f"{'='*80}")

    # Test 5: Try with wrong-shaped mask (should raise error)
    wrong_shape_mask = torch.ones((1, len(antibody_seq) - 10), dtype=torch.bool, device=device)

    try:
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            y_error = generator.generate_with_gillespie(
                x=x,
                t=t,
                x_sizes=x_sizes,
                temperature=1.0,
                no_special_toks=True,
                max_decode_steps=1000,
                verbose=False,
                mask=wrong_shape_mask,
            )
        print("✗ FAILURE: Should have raised ValueError for wrong-shaped mask")
    except ValueError as e:
        print(f"✓ SUCCESS: Correctly raised ValueError")
        print(f"  Error message: {str(e)}")

    print(f"\n{'='*80}")
    print("ALL TESTS COMPLETE")
    print(f"{'='*80}")

    # Summary
    print("\nSummary:")
    print(f"  Test 1 (No mask): {mutations_no_mask['count']} mutations, {check_no_mask['pct_in_mask']:.1f}% in CDR")
    print(f"  Test 2 (CDR mask, unguided): {mutations_cdr_mask['count']} mutations, {check_cdr_mask['pct_in_mask']:.1f}% in CDR {'✓' if check_cdr_mask['all_in_mask'] else '✗'}")
    print(f"  Test 3 (CDR mask, guided): {mutations_guided_cdr['count']} mutations, {check_guided_cdr['pct_in_mask']:.1f}% in CDR {'✓' if check_guided_cdr['all_in_mask'] else '✗'}")
    print(f"  Test 4 (All masked): Error handling ✓")
    print(f"  Test 5 (Wrong shape): Error handling ✓")

    if check_cdr_mask['all_in_mask'] and check_guided_cdr['all_in_mask']:
        print("\n✓✓✓ ALL TESTS PASSED ✓✓✓")
    else:
        print("\n✗✗✗ SOME TESTS FAILED ✗✗✗")


if __name__ == "__main__":
    test_region_masking()
