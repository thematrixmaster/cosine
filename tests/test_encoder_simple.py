"""
Simple test suite to verify MultiSequenceEncoderBlock basic functionality.
"""

import pytest
import torch

from peint.models.nets.transformer import MultiSequenceEncoderBlock


class TestMultiSequenceEncoderBlockSimple:
    """Simple tests for MultiSequenceEncoderBlock."""

    @pytest.fixture
    def basic_encoder(self):
        """Create a basic encoder for testing."""
        return MultiSequenceEncoderBlock(
            attention_heads=4,
            embed_dim=64,
            ffn_embed_dim=256,
            dropout_p=0.0,
            layer_idx=0,
            self_attn_type="intra_inter",
        )

    @pytest.fixture
    def simple_test_data(self):
        """Create simple test data."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        batch_size = 2
        seq_len = 20
        embed_dim = 64

        # Create random input embeddings
        x = torch.randn(batch_size, seq_len, embed_dim, device=device)

        # Create attention mask for two chains
        attn_mask = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)
        chain1_len = seq_len // 2
        chain2_len = seq_len - chain1_len
        attn_mask[:, 0] = chain1_len
        attn_mask[:, 1] = chain2_len

        return x, attn_mask

    def test_basic_forward_pass(self, basic_encoder, simple_test_data):
        """Test basic forward pass functionality."""
        x, attn_mask = simple_test_data
        device = x.device

        encoder = basic_encoder.to(device)

        # Test forward pass
        encoder.eval()
        with torch.no_grad():
            output = encoder(x, attn_mask)

        # Verify output properties
        assert output.shape == x.shape
        assert output.device == x.device
        assert not torch.isnan(output).any()

    def test_encoder_preserves_shape(self, basic_encoder):
        """Test that encoder preserves input shape."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        encoder = basic_encoder.to(device)

        # Test different input shapes
        test_shapes = [
            (1, 10, 64),
            (2, 20, 64),
            (4, 50, 64),
        ]

        for batch_size, seq_len, embed_dim in test_shapes:
            x = torch.randn(batch_size, seq_len, embed_dim, device=device)
            attn_mask = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)
            attn_mask[:, 0] = seq_len // 2
            attn_mask[:, 1] = seq_len - seq_len // 2

            encoder.eval()
            with torch.no_grad():
                output = encoder(x, attn_mask)

            assert output.shape == x.shape

    def test_training_vs_eval_mode(self, basic_encoder, simple_test_data):
        """Test that encoder behaves differently in training vs eval mode."""
        x, attn_mask = simple_test_data
        device = x.device
        encoder = basic_encoder.to(device)

        # Get output in eval mode
        encoder.eval()
        with torch.no_grad():
            eval_output = encoder(x, attn_mask)

        # Get output in training mode
        encoder.train()
        with torch.no_grad():
            train_output = encoder(x, attn_mask)

        # Outputs should be the same when dropout_p=0.0
        assert torch.allclose(eval_output, train_output, atol=1e-6)

    @pytest.mark.parametrize("attention_type", ["full", "intra_only", "intra_inter"])
    def test_all_attention_types_work(self, attention_type, simple_test_data):
        """Test that all attention types can be instantiated and run."""
        x, attn_mask = simple_test_data
        device = x.device

        encoder = MultiSequenceEncoderBlock(
            attention_heads=4,
            embed_dim=64,
            ffn_embed_dim=256,
            dropout_p=0.0,
            layer_idx=0,
            self_attn_type=attention_type,
        ).to(device)

        encoder.eval()
        with torch.no_grad():
            output = encoder(x, attn_mask)

        assert output.shape == x.shape
        assert not torch.isnan(output).any()

    def test_encoder_with_zero_attention_mask(self, basic_encoder):
        """Test encoder behavior with zero attention mask."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        encoder = basic_encoder.to(device)

        batch_size, seq_len, embed_dim = 2, 20, 64
        x = torch.randn(batch_size, seq_len, embed_dim, device=device)

        # Zero attention mask (no valid sequences)
        attn_mask = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)

        encoder.eval()
        with torch.no_grad():
            output = encoder(x, attn_mask)

        # Should still produce output without errors
        assert output.shape == x.shape
        assert not torch.isnan(output).any()

    def test_encoder_parameter_count(self, basic_encoder):
        """Test that encoder has reasonable number of parameters."""
        total_params = sum(p.numel() for p in basic_encoder.parameters())
        trainable_params = sum(p.numel() for p in basic_encoder.parameters() if p.requires_grad)

        # Should have a reasonable number of parameters
        assert total_params > 1000  # At least 1K parameters
        assert total_params < 1_000_000  # Less than 1M parameters for this small model
        assert trainable_params == total_params  # All parameters should be trainable by default

    def test_encoder_device_transfer(self, basic_encoder, simple_test_data):
        """Test that encoder can be moved between devices."""
        x, attn_mask = simple_test_data

        # Test CPU
        encoder_cpu = basic_encoder.to("cpu")
        x_cpu = x.to("cpu")
        attn_mask_cpu = attn_mask.to("cpu")

        encoder_cpu.eval()
        with torch.no_grad():
            output_cpu = encoder_cpu(x_cpu, attn_mask_cpu)

        assert output_cpu.device.type == "cpu"
        assert not torch.isnan(output_cpu).any()

        # Test CUDA if available
        if torch.cuda.is_available():
            encoder_cuda = basic_encoder.to("cuda")
            x_cuda = x.to("cuda")
            attn_mask_cuda = attn_mask.to("cuda")

            encoder_cuda.eval()
            with torch.no_grad():
                output_cuda = encoder_cuda(x_cuda, attn_mask_cuda)

            assert output_cuda.device.type == "cuda"
            assert not torch.isnan(output_cuda).any()

            # Results should be similar between devices
            assert torch.allclose(output_cpu, output_cuda.cpu(), atol=1e-5)
