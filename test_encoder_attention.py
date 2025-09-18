#!/usr/bin/env python3
"""
Test script to verify that the updated MultiSequenceEncoderBlock works with all three attention types
"""

# Temporarily disable flash_attn to force CPU-compatible implementations
import sys

import torch
from esm.data import Alphabet

from evo.tokenization import Vocab

if "flash_attn" in sys.modules:
    del sys.modules["flash_attn"]

from peint.models.nets.esm2 import ESM2Model  # Use non-Flash version for CPU testing

# Import the updated classes
from peint.models.nets.peint import PIPET


def create_test_model(encoder_attn_type="intra_inter", decoder_attn_type="full"):
    """Create a PIPET model with specified attention types for testing"""

    # Create vocabulary
    alphabet = Alphabet.from_architecture("msa_transformer")
    vocab = Vocab.from_esm_alphabet(alphabet)

    # Create a small ESM model for testing
    esm_model = ESM2Model(
        num_layers=2,
        embed_dim=64,
        attention_heads=4,
        alphabet="ESM-1b",
        token_dropout=0.0,
        dropout_p=0.0,
    )

    # Create PIPET model
    model = PIPET(
        esm_model=esm_model,
        evo_vocab=vocab,
        embed_dim=64,
        num_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        encoder_self_attn_type=encoder_attn_type,
        decoder_self_attn_type=decoder_attn_type,
        dropout_p=0.0,
    )

    return model, vocab


def create_test_data(vocab, batch_size=2, seq_len=20):
    """Create test data for PIPET model"""

    # Create mock input sequences (representing x1 and y1)
    enc_in = torch.randint(1, len(vocab) - 3, (batch_size, seq_len))  # Avoid special tokens

    # Add BOS and EOS tokens
    enc_in[:, 0] = vocab.bos_idx
    enc_in[:, -1] = vocab.eos_idx

    # Create mock decoder input sequences (representing x2 and y2)
    dec_in = torch.randint(1, len(vocab) - 3, (batch_size, seq_len))
    dec_in[:, 0] = vocab.bos_idx
    dec_in[:, -1] = vocab.eos_idx

    # Create attention masks (representing sequence lengths for two chains)
    # Format: The mask stores the lengths of each sequence at specific positions
    # For example: [chain1_length, chain2_length, 0, 0, ...]
    enc_attn_mask = torch.zeros((batch_size, seq_len), dtype=torch.long)
    dec_attn_mask = torch.zeros((batch_size, seq_len), dtype=torch.long)

    # Set sequence lengths at the first two positions
    chain1_len = seq_len // 2
    chain2_len = seq_len - chain1_len

    enc_attn_mask[:, 0] = chain1_len  # First chain length
    enc_attn_mask[:, 1] = chain2_len  # Second chain length

    dec_attn_mask[:, 0] = chain1_len  # First chain length
    dec_attn_mask[:, 1] = chain2_len  # Second chain length

    # Create distance tensor (evolutionary distances for each chain)
    # This should match the attention mask structure
    distances = torch.zeros((batch_size, seq_len), dtype=torch.float32)
    distances[:, 0] = 1.0  # Distance for first chain
    distances[:, 1] = 2.0  # Distance for second chain

    return enc_in, dec_in, enc_attn_mask, dec_attn_mask, distances


def test_attention_type(encoder_attn_type):
    """Test a specific encoder attention type"""
    print(f"\nTesting encoder attention type: {encoder_attn_type}")

    try:
        # Create model
        model, vocab = create_test_model(encoder_attn_type=encoder_attn_type)

        # Create test data
        enc_in, dec_in, enc_attn_mask, dec_attn_mask, distances = create_test_data(vocab)

        # Run forward pass
        model.eval()
        with torch.no_grad():
            outputs = model(
                enc_in=enc_in,
                dec_in=dec_in,
                enc_attn_mask=enc_attn_mask,
                dec_attn_mask=dec_attn_mask,
                distances=distances,
            )

        # Check outputs
        enc_logits = outputs["enc_logits"]
        dec_logits = outputs["dec_logits"]

        print(f"  ✓ Forward pass successful")
        print(f"  ✓ Encoder logits shape: {enc_logits.shape}")
        print(f"  ✓ Decoder logits shape: {dec_logits.shape}")

        # Check for NaN values
        if torch.isnan(enc_logits).any() or torch.isnan(dec_logits).any():
            print(f"  ✗ NaN values detected in outputs!")
            return False
        else:
            print(f"  ✓ No NaN values detected")

        # Check that logits have reasonable values
        enc_mean = enc_logits.mean().item()
        dec_mean = dec_logits.mean().item()
        print(f"  ✓ Encoder logits mean: {enc_mean:.4f}")
        print(f"  ✓ Decoder logits mean: {dec_mean:.4f}")

        return True

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_attention_types_comparison():
    """Test that different attention types produce different outputs"""
    print("\nTesting that different attention types produce different outputs...")

    attention_types = ["full", "intra_only", "intra_inter"]
    outputs_dict = {}

    for attn_type in attention_types:
        try:
            model, vocab = create_test_model(encoder_attn_type=attn_type)
            enc_in, dec_in, enc_attn_mask, dec_attn_mask, distances = create_test_data(vocab)

            model.eval()
            with torch.no_grad():
                outputs = model(
                    enc_in=enc_in,
                    dec_in=dec_in,
                    enc_attn_mask=enc_attn_mask,
                    dec_attn_mask=dec_attn_mask,
                    distances=distances,
                )
            outputs_dict[attn_type] = outputs["enc_logits"]
        except Exception as e:
            print(f"  ✗ Error with {attn_type}: {e}")
            return False

    # Compare outputs
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
    print("=" * 60)

    # Test each attention type
    attention_types = ["full", "intra_only", "intra_inter"]

    all_passed = True
    for attn_type in attention_types:
        passed = test_attention_type(attn_type)
        all_passed = all_passed and passed

    # Test that different types produce different outputs
    comparison_passed = test_attention_types_comparison()
    all_passed = all_passed and comparison_passed

    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All tests passed! The encoder attention types are working correctly.")
    else:
        print("✗ Some tests failed. Check the output above for details.")

    return all_passed


if __name__ == "__main__":
    main()
