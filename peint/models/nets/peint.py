from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from esm.model.esm2 import ESM2
from esm.modules import RobertaLMHead
from torch import Tensor

from evo.tokenization import Vocab
from peint.data.utils import RESIDUES
from peint.models.nets.esm2 import ESM2Flash, ESM2Model
from peint.models.nets.transformer import (
    ESMDecoderBlock,
    ESMEncoderBlock,
    FlashMHADecoderBlock,
    FlashMHAEncoderBlock,
    GeometricTimeEmbedder,
    MultiSequenceDecoderBlock,
    MultiSequenceEncoderBlock,
)
from peint.models.nets.utils import _create_sequence_mask, _expand_distances_to_seqlen

STANDARD_STATES = RESIDUES + ["<eos>"]

try:
    import flash_attn  # noqa

    ESM2Wrapper = ESM2Flash
    EncoderBlock = FlashMHAEncoderBlock
    DecoderBlock = FlashMHADecoderBlock
except ImportError:
    ESM2Wrapper = ESM2Model
    EncoderBlock = ESMEncoderBlock
    DecoderBlock = ESMDecoderBlock


def sampling_function(logits: Tensor, p=0.9, argmax_sample=False):
    """
    Perform top p sampling on the given logits.

    Args:
    logits (torch.Tensor): Logits of shape [batch_size, vocab_size]
    p (float): Nucleus sampling parameter, default is 0.9

    Returns:
    torch.Tensor: Sampled token indices of shape [batch_size, 1]
    """

    probs = nn.functional.softmax(logits, dim=-1)

    if argmax_sample or p == 0.0:
        return probs.argmax(-1, keepdim=True)

    # If p == 1, perform standard sampling
    if p >= 1.0:
        return torch.multinomial(probs, 1)

    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    nucleus = cumulative_probs < p

    # Create a mask for the nucleus
    nucleus_mask = nucleus.clone()
    nucleus_mask[:, 1:] = nucleus[:, :-1]
    nucleus_mask[:, 0] = True
    sorted_probs = sorted_probs.masked_fill(~nucleus_mask, 0)

    # Redistribute the probabilities
    sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
    sampled_indices = torch.multinomial(sorted_probs, 1)
    next_tok = torch.gather(sorted_indices, 1, sampled_indices)

    return next_tok


class PEINT(nn.Module):
    """Unified encoder-decoder transformer for protein evolution modeling.

    This model supports both single-sequence (original PEINT) and multi-sequence (PIPET)
    protein evolution modeling through configurable attention mechanisms and data processing.

    The ESM model encodes sequences, and the final hidden representation goes into an
    encoder/decoder stack. The encoder/decoder stack works as a standard transformer
    but with configurable multi-sequence attention support.

    Key Features:
    - Configurable attention types: "full", "intra_only", "intra_inter"
    - Support for both single and multi-sequence inputs
    - Flexible ESM processing (per-chain or concatenated)
    - Unified size-based mask interface

    Args:
        embed_x_per_chain: If True, process ESM per chain separately (multi-sequence style).
                          If False, process concatenated sequences (single-sequence style).
        encoder_self_attn_type: Type of attention for encoder ("full", "intra_only", "intra_inter")
        decoder_self_attn_type: Type of attention for decoder ("full", "intra_only", "intra_inter")
        chain_break_token: Token used to separate sequences (for multi-sequence inputs)
    """

    def __init__(
        self,
        esm_model: ESM2,
        evo_vocab: Vocab,
        embed_dim: int,
        num_heads: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        max_len=1022,
        dropout_p: float = 0.0,
        use_attention_bias: bool = True,
        finetune_esm: bool = False,
        tie_lm_heads: bool = True,
        # Unified parameters
        embed_x_per_chain: bool = False,
        encoder_self_attn_type: str = "full",
        decoder_self_attn_type: str = "full",
        chain_break_token: str = ".",
    ):
        super(PEINT, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        assert num_encoder_layers >= num_decoder_layers
        self.max_len = max_len

        # Configuration parameters
        self.embed_x_per_chain = embed_x_per_chain
        self.encoder_self_attn_type = encoder_self_attn_type
        self.decoder_self_attn_type = decoder_self_attn_type
        self.chain_break_token = chain_break_token

        self.esm = esm_model
        self.finetune_esm = finetune_esm
        self.tie_lm_heads = tie_lm_heads
        self.vocab = evo_vocab
        self.esm.eval() if not self.finetune_esm else self.esm.train()
        self.esm.requires_grad_(self.finetune_esm)  # freeze the ESM model
        self.dropout_p = dropout_p

        # Loss function
        self.criterion = nn.CrossEntropyLoss(reduction="mean", ignore_index=self.vocab.pad_idx)

        # embedding layer from ESM
        self.embedding = nn.Embedding(len(self.vocab), embed_dim)
        self.embedding.load_state_dict(self.esm.embed_tokens.state_dict())
        self.embedding.requires_grad_(self.finetune_esm)  # freeze the ESM embedding layer

        # time embedding
        self.time_embedding = GeometricTimeEmbedder(frequency_embedding_size=embed_dim)

        # Always use multi-sequence blocks (they handle single sequences as special case)
        self.enc_layers = nn.ModuleList(
            [
                MultiSequenceEncoderBlock(
                    attention_heads=self.num_heads,
                    embed_dim=self.embed_dim,
                    ffn_embed_dim=4 * self.embed_dim,
                    dropout_p=self.dropout_p,
                    layer_idx=l,
                    self_attn_type=self.encoder_self_attn_type,
                )
                for l in range(num_encoder_layers)
            ]
        )

        self.dec_layers = nn.ModuleList(
            [
                MultiSequenceDecoderBlock(
                    attention_heads=self.num_heads,
                    embed_dim=self.embed_dim,
                    ffn_embed_dim=4 * self.embed_dim,
                    dropout_p=self.dropout_p,
                    layer_idx=l,
                    self_attn_type=self.decoder_self_attn_type,
                )
                for l in range(num_decoder_layers)
            ]
        )

        self.enc_lm_head = RobertaLMHead(
            embed_dim=self.embed_dim,
            output_dim=len(self.vocab),
            weight=self.embedding.weight,
        )
        self.enc_lm_head.load_state_dict(self.esm.lm_head.state_dict())
        self.enc_lm_head.requires_grad_(self.finetune_esm)

        if self.tie_lm_heads:
            self.dec_lm_head = self.enc_lm_head  # use same lm head for encoder and decoder
        else:
            self.dec_lm_head = RobertaLMHead(
                embed_dim=self.embed_dim,
                output_dim=len(self.vocab),
                weight=self.embedding.weight,
            )
            self.dec_lm_head.load_state_dict(self.esm.lm_head.state_dict())
            self.dec_lm_head.requires_grad_(self.finetune_esm)

    def _compute_language_model_representations(self, x: Tensor, x_sizes: Tensor) -> Tensor:
        """
        Compute ESM representations using unified size-based interface.

        Args:
            x: Input sequences of shape (batch_size, seq_len)
            x_sizes: Sequence sizes (batch_size, num_sequences)

        Returns:
            ESM embeddings of shape (batch_size, seq_len, embed_dim)
        """
        if self.embed_x_per_chain:
            # Multi-sequence processing: compute ESM per chain
            return self._compute_language_model_representations_per_chain(x, x_sizes)
        else:
            # Single-sequence processing: process concatenated sequence
            res = self.esm(x, repr_layers=[self.esm.num_layers], need_head_weights=False)
            esm_s = res["representations"][self.esm.num_layers]
            return esm_s

    def _compute_language_model_representations_per_chain(
        self, x: Tensor, attn_mask_in_length: Tensor
    ):
        """
        Compute ESM embedding separately for each sequence in the batch (multi-sequence style)
        """
        bsz, seq_len = x.shape
        num_sequences = (attn_mask_in_length > 0).sum(dim=1)

        # Assert that all batch elements have the same number of sequences
        assert torch.all(
            num_sequences == num_sequences[0]
        ), "All batch elements must have the same number of sequences"

        num_sequences = num_sequences[0].item()
        combined_embedding = torch.zeros(
            (bsz, seq_len, self.embed_dim), dtype=torch.float, device=x.device
        )

        for seq_idx in range(num_sequences):
            # Shape (B, L), True for the residues of `seq_idx`th sequence of each batch element
            seq_mask = _create_sequence_mask(attn_mask_in_length, sequence_idx=seq_idx)
            # Create a copy of the input to avoid modifying the original
            x_seq = x.clone()
            # Set padding for all other positions
            x_seq.masked_fill_(~seq_mask, self.vocab.pad_idx)

            if (x_seq == self.vocab.pad_idx).all():
                # For a normal use case we don't need this check
                # But for ablation, we might mask out one of the sequence
                # so at least for some seq_idx, x_seq would have only paddings
                # in that case ESM will give nans
                continue

            # Get ESM representation for this sequence
            res = self.esm(x_seq, repr_layers=[self.esm.num_layers], need_head_weights=False)
            esm_s = res["representations"][self.esm.num_layers]

            # Add to combined embedding at the correct positions
            combined_embedding = combined_embedding + esm_s * seq_mask.unsqueeze(-1).float()

        return combined_embedding

    def forward(
        self, x: Tensor, y: Tensor, t: Tensor, x_sizes: Tensor, y_sizes: Tensor
    ) -> Dict[str, Tensor]:
        """
        Unified forward pass using standardized size-based interface.

        Args:
            x: Encoder input sequences (batch_size, seq_len)
            y: Decoder input sequences (batch_size, seq_len)
            t: Time values (batch_size,) or (batch_size, num_sequences)
            x_sizes: Sequence sizes for encoder (batch_size, num_sequences)
            y_sizes: Sequence sizes for decoder (batch_size, num_sequences)

        Returns:
            Dict with 'enc_logits' and 'dec_logits' keys
        """
        # Embed decoder input with time
        h_y = self.embedding(y)

        # Position-specific time embedding using distances
        distances_expanded = _expand_distances_to_seqlen(
            distances_in_length=t, attn_mask_in_length=y_sizes
        )
        ht = self.time_embedding(distances_expanded)
        h_y = h_y + ht

        # Get ESM representations
        h_x = self._compute_language_model_representations(x, x_sizes)

        # Encoder-decoder forward pass
        for i, enc_layer in enumerate(self.enc_layers):
            # Multi-sequence encoder
            h_x = enc_layer(h_x, x_sizes)

            # Decoder layers (with encoder-decoder alignment)
            if self.num_decoder_layers - self.num_encoder_layers + i >= 0:
                idx = self.num_decoder_layers - self.num_encoder_layers + i
                dec_layer = self.dec_layers[idx]
                # Multi-sequence decoder
                h_y, _ = dec_layer(h_y, h_x, y_sizes, x_sizes)

        # Generate logits
        x_logits = self.enc_lm_head(h_x)
        y_logits = self.dec_lm_head(h_y)

        if torch.any(torch.isnan(x_logits)) or torch.any(torch.isnan(y_logits)):
            raise ValueError("NaN detected in logits")

        # Always return dictionary format
        return {"enc_logits": x_logits, "dec_logits": y_logits}

    def decode_sequences(self, decoded: torch.Tensor) -> List[str]:
        """Turns a batch of decoded sequences into a list of strings

        decoded: [B, L] tensor of decoded sequences
        """
        inv_vocab = {v: k for k, v in self.vocab.to_dict().items()}
        output_sequences = []
        for seq in decoded:
            decoded_str = "".join([inv_vocab.get(p.item()) for p in seq[1:]])  # remove cls
            eos_idx = decoded_str.find("<eos>")
            if eos_idx != -1:
                decoded_str = decoded_str[:eos_idx]
            output_sequences.append(decoded_str)
        return output_sequences

    def _precompute_encoder_states(
        self, x: torch.Tensor, x_sizes: torch.Tensor, return_logits=False
    ):
        """Precompute encoder states that don't change during generation"""
        # Get ESM representations
        h_x = self._compute_language_model_representations(x, x_sizes)
        h_x_cached = []

        # Process through encoder layers
        for enc_layer in self.enc_layers:
            h_x = enc_layer(h_x, x_sizes)
            h_x_cached.append(h_x)

        if return_logits:
            x_logits = self.enc_lm_head(h_x)
            return x_logits, h_x_cached

        return h_x_cached

    def _decode_with_cached_encoder(
        self,
        y: torch.Tensor,
        t: torch.Tensor,
        h_x_cached: List[torch.Tensor],
        y_sizes: torch.Tensor,
        x_sizes: torch.Tensor,
        return_hidden_states=False,
    ) -> torch.Tensor:
        """Decode using pre-cached encoder states"""
        # Embed y with time
        h_y = self.embedding(y)

        # Position-specific time embedding using distances
        distances_expanded = _expand_distances_to_seqlen(
            distances_in_length=t, attn_mask_in_length=y_sizes
        )
        ht = self.time_embedding(distances_expanded)
        h_y = h_y + ht
        h_y_cached = []

        # Only run decoder layers that need cross-attention
        for i in range(self.num_decoder_layers):
            idx = i + (self.num_encoder_layers - self.num_decoder_layers)
            dec_layer = self.dec_layers[i]
            h_x = h_x_cached[idx]
            h_y_cached.append(h_y)
            h_y, _ = dec_layer(h_y, h_x, y_sizes, x_sizes)

        if return_hidden_states:
            return self.dec_lm_head(h_y), h_y_cached

        return self.dec_lm_head(h_y)

    @torch.no_grad()
    def generate(
        self,
        x: torch.Tensor,  # <bos>x<eos> format
        t: torch.Tensor,
        max_decode_steps: int,
        device: torch.device,
        temperature: float = 1.0,
        p: float = 1.0,
        x_sizes: torch.Tensor = None,
    ) -> torch.Tensor:
        """Generate sequences using nucleus sampling with cached encoder states"""

        batch_size = x.size(0)

        # If x_sizes not provided, assume single sequence per batch element
        if x_sizes is None:
            seq_len = (x != self.vocab.pad_idx).sum(dim=1, keepdim=True)
            x_sizes = seq_len.to(torch.long)

        y_decoded = torch.tensor([self.vocab.bos_idx]).unsqueeze(0).repeat(batch_size, 1).to(device)
        eos_reached = torch.zeros(batch_size, dtype=torch.bool).to(device)

        # Precompute encoder states once
        h_x_cached = self._precompute_encoder_states(x, x_sizes)

        for _ in range(max_decode_steps):
            # Current y_sizes for decoder (single sequence growing)
            y_seq_len = (y_decoded != self.vocab.pad_idx).sum(dim=1, keepdim=True)
            y_sizes = y_seq_len.to(torch.long)

            # Use cached encoder states for decoding
            logits = self._decode_with_cached_encoder(y_decoded, t, h_x_cached, y_sizes, x_sizes)

            logits = logits[:, -1, :] / temperature
            # want to renormalize the probs away from extraneous states, this includes gaps
            zero_idx = torch.tensor(
                [self.vocab.index(t) for t in self.vocab.tokens if t not in STANDARD_STATES]
            )
            logits[..., zero_idx] = -np.inf

            next_tok = sampling_function(logits, p=p)
            next_tok[eos_reached] = self.vocab.pad_idx
            y_decoded = torch.cat([y_decoded, next_tok], dim=1)

            eos_reached |= next_tok.squeeze(-1) == self.vocab.eos_idx
            if eos_reached.all():
                break

        return y_decoded

    @torch.no_grad()
    def perplexity(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,  # assumes <bos>y<eos> format
        batch_size: int = 64,
        x_sizes: torch.Tensor = None,
    ) -> torch.Tensor:
        """Compute perplexity of (y, t) pairs given each x"""

        # If x_sizes not provided, assume single sequence per batch element
        if x_sizes is None:
            seq_len = (x != self.vocab.pad_idx).sum(dim=1, keepdim=True)
            x_sizes = seq_len.to(torch.long)

        # Precompute encoder states once
        h_x_cached = self._precompute_encoder_states(x, x_sizes)

        n_inputs = x.shape[0]
        n_targets = y.shape[0]
        assert y.shape[0] == t.shape[0]
        y_lengths = (y[:, 1:] != self.vocab.pad_idx).sum(dim=-1).float()

        # Initialize result tensor
        all_perplexities = torch.zeros(n_inputs, n_targets, device=x.device)

        for i in range(n_inputs):
            target_nlls = []

            for j in range(0, n_targets, batch_size):
                y_chunk = y[j : j + batch_size]
                t_chunk = t[j : j + batch_size]
                _bs = y_chunk.shape[0]

                # Create y_sizes for this chunk
                y_seq_len = (y_chunk != self.vocab.pad_idx).sum(dim=1, keepdim=True)
                y_sizes = y_seq_len.to(torch.long)

                _h_x_cached = [h[i].unsqueeze(0).expand(_bs, -1, -1) for h in h_x_cached]
                _x_sizes = x_sizes[i : i + 1].expand(_bs, -1)

                y_logits = self._decode_with_cached_encoder(
                    y_chunk, t_chunk, _h_x_cached, y_sizes, _x_sizes
                )

                y_nll = nn.functional.cross_entropy(
                    y_logits[:, :-1, :].transpose(-1, -2),
                    y_chunk[:, 1:],
                    ignore_index=self.vocab.pad_idx,
                    reduction="none",
                )
                y_tgt_mask = y_chunk[:, 1:] != self.vocab.pad_idx
                y_nll = (y_nll * y_tgt_mask.float()).sum(-1)
                target_nlls.append(y_nll)

            # Concatenate likelihoods for this input
            input_nlls = torch.cat(target_nlls, dim=0)
            # Compute perplexities for this input against all targets
            all_perplexities[i] = torch.exp(input_nlls / y_lengths)

        return all_perplexities.cpu().transpose(0, 1)  # shape: [n_targets, n_inputs]

    @classmethod
    def from_pretrained_esm2(cls, *args, **kwargs) -> "PEINT":
        """Create a PEINTModule from a pretrained ESM model."""
        import esm

        _esm_model, esm_vocab = esm.pretrained.esm2_t30_150M_UR50D()
        esm_model = ESM2Wrapper(
            num_layers=_esm_model.num_layers,
            embed_dim=_esm_model.embed_dim,
            attention_heads=_esm_model.attention_heads,
            alphabet="ESM-1b",
            token_dropout=_esm_model.token_dropout,
            dropout_p=kwargs.get("dropout_p", 0.0),
        )

        ckpt_path = kwargs.pop("ckpt_path", None)
        if ckpt_path is not None:
            state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
            esm_model.load_state_dict(state_dict=state_dict, strict=False)
        else:
            esm_model.load_state_dict(_esm_model.state_dict(), strict=False)
            del _esm_model

        evo_vocab = Vocab.from_esm_alphabet(esm_vocab)
        return cls(esm_model=esm_model, evo_vocab=evo_vocab, *args, **kwargs)
