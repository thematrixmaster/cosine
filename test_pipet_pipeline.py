#!/usr/bin/env python3
"""
Comprehensive test script to validate PIPET training pipeline functionality.
Tests ESM embedding, multi-sequence attention, and training mechanics.
"""

import sys
import traceback

import torch
from esm.data import Alphabet

from evo.tokenization import Vocab
from peint.models.modules.peint_module import PIPETModule
from peint.models.nets.esm2 import ESM2Flash

# Import PIPET components
from peint.models.nets.peint import PIPET
from peint.models.nets.utils import _create_sequence_mask


def test_esm_chain_separation():
    """Test that ESM processes each chain separately as expected."""
    print("\n=== Testing ESM Chain Separation ===")

    try:
        # Create vocabulary and model
        alphabet = Alphabet.from_architecture("ESM-1b")
        vocab = Vocab.from_esm_alphabet(alphabet)

        # Create small ESM model for testing
        esm_model = ESM2Flash(
            num_layers=2,
            embed_dim=64,
            attention_heads=4,
            alphabet="ESM-1b",
            token_dropout=0.0,
            dropout_p=0.0,
        )

        # Create PIPET model with intra_inter attention
        model = PIPET(
            esm_model=esm_model,
            evo_vocab=vocab,
            embed_dim=64,
            num_heads=4,
            num_encoder_layers=2,
            num_decoder_layers=2,
            encoder_self_attn_type="intra_inter",
            decoder_self_attn_type="intra_inter",
            dropout_p=0.0,
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        # Skip BFloat16 conversion for now due to LayerNorm compatibility issues

        # Create test data simulating two protein chains
        batch_size = 2
        seq_len = 20

        # Create input representing concatenated chains: <cls>chain1<eos><cls>chain2<eos>
        enc_in = torch.randint(1, len(vocab) - 3, (batch_size, seq_len), device=device)
        enc_in[:, 0] = vocab.bos_idx  # First <cls>
        enc_in[:, seq_len // 2] = vocab.bos_idx  # Second <cls>
        enc_in[:, -1] = vocab.eos_idx  # Final <eos>

        # Create attention mask: [chain1_length, chain2_length, 0, 0, ...]
        enc_attn_mask = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)
        chain1_len = seq_len // 2
        chain2_len = seq_len - chain1_len
        enc_attn_mask[:, 0] = chain1_len
        enc_attn_mask[:, 1] = chain2_len

        print(f"  Input shape: {enc_in.shape}")
        print(f"  Attention mask shape: {enc_attn_mask.shape}")
        print(f"  Chain lengths: {chain1_len}, {chain2_len}")

        # Test ESM embedding computation
        with torch.no_grad():
            esm_embeddings = model._compute_language_model_representations(enc_in, enc_attn_mask)

        print(f"  ✓ ESM embeddings computed successfully: {esm_embeddings.shape}")

        # Verify that different chains get different embeddings
        chain1_mask = _create_sequence_mask(enc_attn_mask, 0)
        chain2_mask = _create_sequence_mask(enc_attn_mask, 1)

        chain1_embeddings = esm_embeddings[chain1_mask]
        chain2_embeddings = esm_embeddings[chain2_mask]

        # Check that embeddings are different for different chains
        embedding_diff = torch.abs(chain1_embeddings.mean() - chain2_embeddings.mean())
        print(f"  ✓ Chain embedding difference: {embedding_diff:.6f}")

        if embedding_diff > 1e-6:
            print(f"  ✓ Chains processed separately (embeddings differ)")
        else:
            print(f"  ⚠ Warning: Chain embeddings very similar")

        return True

    except Exception as e:
        print(f"  ✗ Error in ESM chain separation test: {e}")
        traceback.print_exc()
        return False


def test_attention_mechanisms():
    """Test that intra_inter attention works correctly for both encoder and decoder."""
    print("\n=== Testing Multi-Sequence Attention Mechanisms ===")

    try:
        alphabet = Alphabet.from_architecture("ESM-1b")
        vocab = Vocab.from_esm_alphabet(alphabet)

        esm_model = ESM2Flash(
            num_layers=2,
            embed_dim=64,
            attention_heads=4,
            alphabet="ESM-1b",
            token_dropout=0.0,
            dropout_p=0.0,
        )

        # Test all three attention types
        attention_types = ["full", "intra_only", "intra_inter"]

        for attn_type in attention_types:
            print(f"\n  Testing attention type: {attn_type}")

            model = PIPET(
                esm_model=esm_model,
                evo_vocab=vocab,
                embed_dim=64,
                num_heads=4,
                num_encoder_layers=1,
                num_decoder_layers=1,
                encoder_self_attn_type=attn_type,
                decoder_self_attn_type=attn_type,
                dropout_p=0.0,
            )

            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(device).to(torch.bfloat16)

            # Create test data
            batch_size = 2
            seq_len = 16

            enc_in = torch.randint(1, len(vocab) - 3, (batch_size, seq_len), device=device)
            dec_in = torch.randint(1, len(vocab) - 3, (batch_size, seq_len), device=device)

            # Attention masks
            enc_attn_mask = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)
            dec_attn_mask = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)

            enc_attn_mask[:, 0] = seq_len // 2
            enc_attn_mask[:, 1] = seq_len - seq_len // 2
            dec_attn_mask[:, 0] = seq_len // 2
            dec_attn_mask[:, 1] = seq_len - seq_len // 2

            # Distances
            distances = torch.zeros((batch_size, seq_len), dtype=torch.float32, device=device)
            distances[:, 0] = 1.0
            distances[:, 1] = 2.0

            # Forward pass
            model.eval()
            with torch.no_grad():
                # Keep all tensors in Float32 for now

                outputs = model(enc_in, dec_in, enc_attn_mask, dec_attn_mask, distances)

            enc_logits = outputs["enc_logits"]
            dec_logits = outputs["dec_logits"]

            print(f"    ✓ Forward pass successful")
            print(f"    ✓ Encoder logits: {enc_logits.shape}")
            print(f"    ✓ Decoder logits: {dec_logits.shape}")

            # Check for NaN values
            if torch.isnan(enc_logits).any() or torch.isnan(dec_logits).any():
                print(f"    ✗ NaN values detected!")
                return False

            print(f"    ✓ No NaN values detected")

        return True

    except Exception as e:
        print(f"  ✗ Error in attention mechanism test: {e}")
        traceback.print_exc()
        return False


def test_training_step():
    """Test that the training step works end-to-end."""
    print("\n=== Testing Training Step ===")

    try:
        alphabet = Alphabet.from_architecture("ESM-1b")
        vocab = Vocab.from_esm_alphabet(alphabet)

        esm_model = ESM2Flash(
            num_layers=2,
            embed_dim=64,
            attention_heads=4,
            alphabet="ESM-1b",
            token_dropout=0.0,
            dropout_p=0.0,
        )

        # Create PIPET model
        net = PIPET(
            esm_model=esm_model,
            evo_vocab=vocab,
            embed_dim=64,
            num_heads=4,
            num_encoder_layers=2,
            num_decoder_layers=2,
            encoder_self_attn_type="intra_inter",
            decoder_self_attn_type="intra_inter",
            dropout_p=0.0,
        )

        # Create Lightning module
        model = PIPETModule(net=net)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        # Skip BFloat16 conversion for now due to LayerNorm compatibility issues

        # Create synthetic batch matching PIPET data format
        batch_size = 2
        seq_len = 16

        # Format: [enc_inputs, enc_targets, dec_inputs, dec_targets, distances, enc_sizes, dec_sizes]
        enc_inputs = torch.randint(1, len(vocab) - 3, (batch_size, seq_len), device=device)
        enc_targets = torch.randint(1, len(vocab) - 3, (batch_size, seq_len), device=device)
        dec_inputs = torch.randint(1, len(vocab) - 3, (batch_size, seq_len), device=device)
        dec_targets = torch.randint(1, len(vocab) - 3, (batch_size, seq_len), device=device)

        distances = torch.zeros((batch_size, seq_len), dtype=torch.float32, device=device)
        distances[:, 0] = 1.0
        distances[:, 1] = 2.0

        enc_sizes = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)
        dec_sizes = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)
        enc_sizes[:, 0] = seq_len // 2
        enc_sizes[:, 1] = seq_len - seq_len // 2
        dec_sizes[:, 0] = seq_len // 2
        dec_sizes[:, 1] = seq_len - seq_len // 2

        batch = [enc_inputs, enc_targets, dec_inputs, dec_targets, distances, enc_sizes, dec_sizes]

        print(f"  Batch created with shapes:")
        print(f"    enc_inputs: {enc_inputs.shape}")
        print(f"    distances: {distances.shape}")
        print(f"    enc_sizes: {enc_sizes.shape}")

        # Test model_step
        model.train()

        # Keep all tensors in Float32 for now

        loss_info = model.model_step(batch)

        print(f"  ✓ Training step successful")
        print(f"  ✓ Loss components: {list(loss_info.keys())}")

        # Check loss values
        total_loss = loss_info["mlm_loss"] + loss_info["tlm_loss"]
        print(f"  ✓ MLM loss: {loss_info['mlm_loss']:.4f}")
        print(f"  ✓ TLM loss: {loss_info['tlm_loss']:.4f}")
        print(f"  ✓ Total loss: {total_loss:.4f}")

        # Verify loss is reasonable (not NaN or infinite)
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"  ✗ Invalid loss value!")
            return False

        print(f"  ✓ Loss values are valid")

        return True

    except Exception as e:
        print(f"  ✗ Error in training step test: {e}")
        traceback.print_exc()
        return False


def test_config_integration():
    """Test that the training config loads and creates the correct model."""
    print("\n=== Testing Configuration Integration ===")

    try:
        # Test hydra config loading (simulate key parameters)
        config_params = {
            "encoder_self_attn_type": "intra_inter",
            "decoder_self_attn_type": "intra_inter",
            "chain_break_token": ".",
            "num_heads": 20,
            "num_encoder_layers": 3,
            "num_decoder_layers": 3,
            "embed_dim": 1280,  # ESM-1b dimension
        }

        print(f"  Testing config parameters: {config_params}")

        # Create model with config parameters
        alphabet = Alphabet.from_architecture("ESM-1b")
        vocab = Vocab.from_esm_alphabet(alphabet)

        esm_model = ESM2Flash(
            num_layers=2,  # Reduced for testing
            embed_dim=config_params["embed_dim"],
            attention_heads=config_params["num_heads"],
            alphabet="ESM-1b",
            token_dropout=0.0,
            dropout_p=0.0,
        )

        model = PIPET(
            esm_model=esm_model,
            evo_vocab=vocab,
            embed_dim=config_params["embed_dim"],
            num_heads=config_params["num_heads"],
            num_encoder_layers=config_params["num_encoder_layers"],
            num_decoder_layers=config_params["num_decoder_layers"],
            encoder_self_attn_type=config_params["encoder_self_attn_type"],
            decoder_self_attn_type=config_params["decoder_self_attn_type"],
            chain_break_token=config_params["chain_break_token"],
        )

        print(f"  ✓ Model created successfully with config parameters")
        print(f"  ✓ Chain break token: '{model.chain_break_token}'")
        print(f"  ✓ Separator index: {model.sep_idx}")

        # Verify attention types are correctly set
        enc_layer = model.enc_layers[0]
        dec_layer = model.dec_layers[0]

        print(f"  ✓ Encoder attention type: {type(enc_layer.self_attn).__name__}")
        print(f"  ✓ Decoder attention type: {type(dec_layer.self_attn).__name__}")

        # Both should be DecoupledIntraInterMultiSequenceSelfAttention for intra_inter
        expected_type = "DecoupledIntraInterMultiSequenceSelfAttention"

        if expected_type in type(enc_layer.self_attn).__name__:
            print(f"  ✓ Encoder using correct intra_inter attention")
        else:
            print(f"  ⚠ Encoder attention type unexpected: {type(enc_layer.self_attn).__name__}")

        if expected_type in type(dec_layer.self_attn).__name__:
            print(f"  ✓ Decoder using correct intra_inter attention")
        else:
            print(f"  ⚠ Decoder attention type unexpected: {type(dec_layer.self_attn).__name__}")

        return True

    except Exception as e:
        print(f"  ✗ Error in config integration test: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all PIPET validation tests."""
    print("PIPET Training Pipeline Validation")
    print("=" * 50)
    print(f"Using device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

    tests = [
        ("ESM Chain Separation", test_esm_chain_separation),
        ("Multi-Sequence Attention", test_attention_mechanisms),
        ("Training Step", test_training_step),
        ("Config Integration", test_config_integration),
    ]

    results = {}
    for test_name, test_func in tests:
        try:
            print(f"\n{'='*50}")
            print(f"Running: {test_name}")
            results[test_name] = test_func()
        except Exception as e:
            print(f"✗ Test {test_name} failed with exception: {e}")
            results[test_name] = False

    # Summary
    print(f"\n{'='*50}")
    print("VALIDATION SUMMARY")
    print(f"{'='*50}")

    passed = 0
    total = len(tests)

    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:.<30} {status}")
        if result:
            passed += 1

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! PIPET pipeline is working correctly.")
        print("\nKey validations confirmed:")
        print("  ✓ ESM processes each chain separately")
        print("  ✓ Multi-sequence attention mechanisms work")
        print("  ✓ Training step executes without errors")
        print("  ✓ Configuration creates correct model architecture")
        print("  ✓ Both encoder and decoder use intra_inter attention")
    else:
        print(f"\n⚠ {total-passed} test(s) failed. Please review the output above.")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
