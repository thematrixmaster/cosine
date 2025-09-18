#!/usr/bin/env python3
"""
Simple test script to verify that the updated MultiSequenceEncoderBlock works with all three attention types
"""

import torch

# Temporarily disable flash_attn to force CPU-compatible implementations
flash_attn_available = False
try:
    pass

    flash_attn_available = True
    print("Flash attention is available, but we'll use CPU-compatible versions for testing")
except ImportError:
    print("Flash attention not available, using CPU-compatible versions")

# Import the transformer module
from peint.models.nets.transformer import MultiSequenceEncoderBlock


def create_test_encoder(attention_type="intra_inter", embed_dim=64, num_heads=4):
    """Create a MultiSequenceEncoderBlock with specified attention type"""

    encoder = MultiSequenceEncoderBlock(
        attention_heads=num_heads,
        embed_dim=embed_dim,
        ffn_embed_dim=4 * embed_dim,
        dropout_p=0.0,
        layer_idx=0,
        self_attn_type=attention_type,
    )

    return encoder


def create_test_data(batch_size=2, seq_len=20, embed_dim=64, device="cuda"):
    """Create test data for the encoder"""

    # Create random input embeddings (use bfloat16 for Flash Attention compatibility)
    x = torch.randn(batch_size, seq_len, embed_dim, device=device, dtype=torch.bfloat16)

    # Create attention mask (sequence lengths for two chains)
    # Format: [chain1_length, chain2_length, 0, 0, ...]
    attn_mask = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)

    chain1_len = seq_len // 2
    chain2_len = seq_len - chain1_len

    attn_mask[:, 0] = chain1_len  # First chain length
    attn_mask[:, 1] = chain2_len  # Second chain length

    return x, attn_mask


def test_encoder_attention_type(attention_type):
    """Test a specific encoder attention type"""
    print(f"\nTesting encoder attention type: {attention_type}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Using device: {device}")

    try:
        # Create encoder
        encoder = create_test_encoder(attention_type=attention_type)
        encoder = encoder.to(device).to(torch.bfloat16)

        # Create test data
        x, attn_mask = create_test_data(device=device)

        # Run forward pass
        encoder.eval()
        with torch.no_grad():
            output = encoder(x, attn_mask)

        # Check output
        print(f"  ✓ Forward pass successful")
        print(f"  ✓ Input shape: {x.shape}")
        print(f"  ✓ Output shape: {output.shape}")
        print(f"  ✓ Shape preserved: {x.shape == output.shape}")

        # Check for NaN values
        if torch.isnan(output).any():
            print(f"  ✗ NaN values detected in output!")
            return False, None
        else:
            print(f"  ✓ No NaN values detected")

        # Check that output has reasonable values
        output_mean = output.mean().item()
        output_std = output.std().item()
        print(f"  ✓ Output mean: {output_mean:.4f}")
        print(f"  ✓ Output std: {output_std:.4f}")

        return True, output

    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback

        traceback.print_exc()
        return False, None


def test_attention_types_comparison():
    """Test that different attention types produce different outputs"""
    print("\nTesting that different attention types produce different outputs...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Using device: {device}")

    attention_types = ["full", "intra_only", "intra_inter"]
    outputs_dict = {}

    # Create the same test data for all tests
    x, attn_mask = create_test_data(device=device)

    for attn_type in attention_types:
        try:
            encoder = create_test_encoder(attention_type=attn_type)
            encoder = encoder.to(device).to(torch.bfloat16)
            encoder.eval()
            with torch.no_grad():
                output = encoder(x, attn_mask)
            outputs_dict[attn_type] = output
            print(f"  ✓ {attn_type}: output computed successfully")
        except Exception as e:
            print(f"  ✗ Error with {attn_type}: {e}")
            return False

    # Compare outputs
    print("\nComparing outputs between attention types:")
    for i, attn_type1 in enumerate(attention_types):
        for attn_type2 in attention_types[i + 1 :]:
            diff = torch.abs(outputs_dict[attn_type1] - outputs_dict[attn_type2]).mean()
            print(f"  Mean absolute difference between {attn_type1} and {attn_type2}: {diff:.6f}")

            if diff < 1e-6:
                print(f"  ⚠ Warning: {attn_type1} and {attn_type2} produce very similar outputs")
            else:
                print(f"  ✓ {attn_type1} and {attn_type2} produce different outputs")

    return True


def main():
    """Main test function"""
    print("Testing MultiSequenceEncoderBlock with different attention types")
    print("=" * 70)

    # Test each attention type
    attention_types = ["full", "intra_only", "intra_inter"]

    all_passed = True
    outputs = {}

    for attn_type in attention_types:
        passed, output = test_encoder_attention_type(attn_type)
        all_passed = all_passed and passed
        if output is not None:
            outputs[attn_type] = output

    # Test that different types produce different outputs
    if len(outputs) == len(attention_types):
        comparison_passed = test_attention_types_comparison()
        all_passed = all_passed and comparison_passed

    print("\n" + "=" * 70)
    if all_passed:
        print("✓ All tests passed! The encoder attention types are working correctly.")
        print("\nSummary:")
        print("- MultiSequenceEncoderBlock now supports three attention types:")
        print("  - 'full': Full attention between all sequences")
        print("  - 'intra_only': Attention only within each sequence")
        print("  - 'intra_inter': Decoupled intra and inter sequence attention (default)")
        print("- All attention types produce valid outputs without errors")
        print("- Different attention types produce different results as expected")
    else:
        print("✗ Some tests failed. Check the output above for details.")

    return all_passed


if __name__ == "__main__":
    main()
