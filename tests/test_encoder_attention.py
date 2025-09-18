"""
Test suite for MultiSequenceEncoderBlock with different attention types.
"""

import pytest
import torch

from peint.models.nets.transformer import MultiSequenceEncoderBlock


class TestMultiSequenceEncoderBlock:
    """Test MultiSequenceEncoderBlock with different attention types."""

    @pytest.fixture(params=["full", "intra_only", "intra_inter"])
    def attention_type(self, request):
        """Parametrized attention types for testing."""
        return request.param

    @pytest.fixture
    def encoder(self, attention_type):
        """Create a MultiSequenceEncoderBlock with specified attention type."""
        return MultiSequenceEncoderBlock(
            attention_heads=4,
            embed_dim=64,
            ffn_embed_dim=256,
            dropout_p=0.0,
            layer_idx=0,
            self_attn_type=attention_type,
        )

    @pytest.fixture
    def test_data(self):
        """Create test data for the encoder."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        batch_size = 2
        seq_len = 20
        embed_dim = 64

        # Create random input embeddings
        x = torch.randn(batch_size, seq_len, embed_dim, device=device)

        # Create attention mask (sequence lengths for two chains)
        attn_mask = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)
        chain1_len = seq_len // 2
        chain2_len = seq_len - chain1_len
        attn_mask[:, 0] = chain1_len  # First chain length
        attn_mask[:, 1] = chain2_len  # Second chain length

        return x, attn_mask

    def test_encoder_forward_pass(self, encoder, test_data, attention_type):
        """Test that encoder forward pass works for all attention types."""
        x, attn_mask = test_data
        device = x.device

        encoder = encoder.to(device)

        # Run forward pass
        encoder.eval()
        with torch.no_grad():
            output = encoder(x, attn_mask)

        # Check output shape
        assert (
            output.shape == x.shape
        ), f"Output shape should match input shape for {attention_type}"

        # Check for NaN values
        assert not torch.isnan(output).any(), f"Output should not contain NaN for {attention_type}"

        # Check that output has reasonable values
        output_mean = output.mean().item()
        output_std = output.std().item()

        # Values should be reasonable (not too extreme)
        assert abs(output_mean) < 10, f"Output mean should be reasonable for {attention_type}"
        assert (
            0 < output_std < 10
        ), f"Output std should be positive and reasonable for {attention_type}"

    def test_attention_types_produce_different_outputs(self):
        """Test that different attention types produce different outputs."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        batch_size = 2
        seq_len = 16
        embed_dim = 64

        # Create the same test data for all tests
        x = torch.randn(batch_size, seq_len, embed_dim, device=device)
        attn_mask = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)
        attn_mask[:, 0] = seq_len // 2
        attn_mask[:, 1] = seq_len - seq_len // 2

        attention_types = ["full", "intra_only", "intra_inter"]
        outputs = {}

        # Compute outputs for all attention types
        for attn_type in attention_types:
            encoder = MultiSequenceEncoderBlock(
                attention_heads=4,
                embed_dim=embed_dim,
                ffn_embed_dim=4 * embed_dim,
                dropout_p=0.0,
                layer_idx=0,
                self_attn_type=attn_type,
            ).to(device)

            encoder.eval()
            with torch.no_grad():
                output = encoder(x, attn_mask)
            outputs[attn_type] = output

        # Compare outputs between different attention types
        for i, attn_type1 in enumerate(attention_types):
            for attn_type2 in attention_types[i + 1 :]:
                diff = torch.abs(outputs[attn_type1] - outputs[attn_type2]).mean()

                # Different attention types should produce meaningfully different outputs
                assert (
                    diff > 1e-6
                ), f"{attn_type1} and {attn_type2} should produce different outputs"

    @pytest.mark.parametrize("embed_dim", [32, 64, 128])
    @pytest.mark.parametrize("num_heads", [2, 4, 8])
    def test_different_dimensions(self, embed_dim, num_heads):
        """Test encoder with different embedding dimensions and head counts."""
        if embed_dim % num_heads != 0:
            pytest.skip(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        batch_size = 2
        seq_len = 16

        encoder = MultiSequenceEncoderBlock(
            attention_heads=num_heads,
            embed_dim=embed_dim,
            ffn_embed_dim=4 * embed_dim,
            dropout_p=0.0,
            layer_idx=0,
            self_attn_type="intra_inter",
        ).to(device)

        x = torch.randn(batch_size, seq_len, embed_dim, device=device)
        attn_mask = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)
        attn_mask[:, 0] = seq_len // 2
        attn_mask[:, 1] = seq_len - seq_len // 2

        encoder.eval()
        with torch.no_grad():
            output = encoder(x, attn_mask)

        assert output.shape == (batch_size, seq_len, embed_dim)
        assert not torch.isnan(output).any()

    def test_gradient_flow(self):
        """Test that gradients flow properly through the encoder."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        batch_size = 2
        seq_len = 16
        embed_dim = 64

        encoder = MultiSequenceEncoderBlock(
            attention_heads=4,
            embed_dim=embed_dim,
            ffn_embed_dim=4 * embed_dim,
            dropout_p=0.0,
            layer_idx=0,
            self_attn_type="intra_inter",
        ).to(device)

        x = torch.randn(batch_size, seq_len, embed_dim, device=device, requires_grad=True)
        attn_mask = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)
        attn_mask[:, 0] = seq_len // 2
        attn_mask[:, 1] = seq_len - seq_len // 2

        encoder.train()
        output = encoder(x, attn_mask)
        loss = output.sum()
        loss.backward()

        # Check that gradients are computed
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

        # Check that model parameters have gradients
        for param in encoder.parameters():
            if param.requires_grad:
                assert param.grad is not None
                assert not torch.isnan(param.grad).any()
