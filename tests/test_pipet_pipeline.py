"""
Test suite for PIPET training pipeline functionality.
Tests ESM embedding, multi-sequence attention, and training mechanics using pytest.
"""

import pytest
import torch
from esm.data import Alphabet

from evo.tokenization import Vocab
from peint.models.modules.peint_module import PIPETModule
from peint.models.nets.esm2 import ESM2Flash
from peint.models.nets.peint import PIPET
from peint.models.nets.utils import _create_sequence_mask


class TestPIPETESMChainSeparation:
    """Test ESM chain separation functionality."""

    @pytest.fixture
    def vocab(self):
        """Create vocabulary for testing."""
        alphabet = Alphabet.from_architecture("ESM-1b")
        return Vocab.from_esm_alphabet(alphabet)

    @pytest.fixture
    def esm_model(self):
        """Create small ESM model for testing."""
        return ESM2Flash(
            num_layers=2,
            embed_dim=64,
            attention_heads=4,
            alphabet="ESM-1b",
            token_dropout=0.0,
            dropout_p=0.0,
        )

    @pytest.fixture
    def pipet_model(self, esm_model, vocab):
        """Create PIPET model with intra_inter attention."""
        return PIPET(
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

    def test_esm_chain_separation(self, pipet_model, vocab):
        """Test that ESM processes each chain separately as expected."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = pipet_model.to(device)

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

        # Test ESM embedding computation
        model.eval()
        with torch.no_grad():
            esm_embeddings = model._compute_language_model_representations(enc_in, enc_attn_mask)

        # Verify embeddings have correct shape
        assert esm_embeddings.shape == (batch_size, seq_len, 64)

        # Verify that different chains get different embeddings
        chain1_mask = _create_sequence_mask(enc_attn_mask, 0)
        chain2_mask = _create_sequence_mask(enc_attn_mask, 1)

        chain1_embeddings = esm_embeddings[chain1_mask]
        chain2_embeddings = esm_embeddings[chain2_mask]

        # Check that embeddings are different for different chains
        embedding_diff = torch.abs(chain1_embeddings.mean() - chain2_embeddings.mean())
        assert embedding_diff > 1e-6, "Chains should be processed differently"


class TestPIPETAttentionMechanisms:
    """Test multi-sequence attention mechanisms."""

    @pytest.fixture
    def vocab(self):
        """Create vocabulary for testing."""
        alphabet = Alphabet.from_architecture("ESM-1b")
        return Vocab.from_esm_alphabet(alphabet)

    @pytest.fixture
    def esm_model(self):
        """Create ESM model for testing."""
        return ESM2Flash(
            num_layers=2,
            embed_dim=64,
            attention_heads=4,
            alphabet="ESM-1b",
            token_dropout=0.0,
            dropout_p=0.0,
        )

    @pytest.mark.parametrize("attn_type", ["full", "intra_only", "intra_inter"])
    def test_attention_mechanisms(self, esm_model, vocab, attn_type):
        """Test that attention mechanisms work correctly for all types."""
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
        model = model.to(device)

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
            outputs = model(enc_in, dec_in, enc_attn_mask, dec_attn_mask, distances)

        enc_logits = outputs["enc_logits"]
        dec_logits = outputs["dec_logits"]

        # Verify outputs have correct shapes
        assert enc_logits.shape == (batch_size, seq_len, len(vocab))
        assert dec_logits.shape == (batch_size, seq_len, len(vocab))

        # Check for NaN values
        assert not torch.isnan(enc_logits).any(), "Encoder logits should not contain NaN"
        assert not torch.isnan(dec_logits).any(), "Decoder logits should not contain NaN"


class TestPIPETTrainingStep:
    """Test PIPET training step functionality."""

    @pytest.fixture
    def vocab(self):
        """Create vocabulary for testing."""
        alphabet = Alphabet.from_architecture("ESM-1b")
        return Vocab.from_esm_alphabet(alphabet)

    @pytest.fixture
    def lightning_module(self, vocab):
        """Create PIPET Lightning module for testing."""
        esm_model = ESM2Flash(
            num_layers=2,
            embed_dim=64,
            attention_heads=4,
            alphabet="ESM-1b",
            token_dropout=0.0,
            dropout_p=0.0,
        )

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

        return PIPETModule(net=net)

    def test_training_step(self, lightning_module, vocab):
        """Test that the training step works end-to-end."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = lightning_module.to(device)

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

        # Test model_step
        model.train()
        loss_info = model.model_step(batch)

        # Check loss values
        assert "mlm_loss" in loss_info, "MLM loss should be present"
        assert "tlm_loss" in loss_info, "TLM loss should be present"

        total_loss = loss_info["mlm_loss"] + loss_info["tlm_loss"]

        # Verify loss is reasonable (not NaN or infinite)
        assert not torch.isnan(total_loss), "Total loss should not be NaN"
        assert not torch.isinf(total_loss), "Total loss should not be infinite"
        assert total_loss > 0, "Total loss should be positive"


class TestPIPETConfiguration:
    """Test PIPET configuration integration."""

    def test_config_integration(self):
        """Test that the training config creates the correct model."""
        # Test configuration parameters
        config_params = {
            "encoder_self_attn_type": "intra_inter",
            "decoder_self_attn_type": "intra_inter",
            "chain_break_token": ".",
            "num_heads": 20,
            "num_encoder_layers": 3,
            "num_decoder_layers": 3,
            "embed_dim": 1280,  # ESM-1b dimension
        }

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

        # Verify configuration
        assert model.chain_break_token == config_params["chain_break_token"]
        assert len(model.enc_layers) == config_params["num_encoder_layers"]
        assert len(model.dec_layers) == config_params["num_decoder_layers"]

        # Verify attention types are correctly set
        enc_layer = model.enc_layers[0]
        dec_layer = model.dec_layers[0]

        expected_type = "DecoupledIntraInterMultiSequenceSelfAttention"
        assert expected_type in type(enc_layer.self_attn).__name__
        assert expected_type in type(dec_layer.self_attn).__name__
