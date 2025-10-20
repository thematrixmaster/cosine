from abc import ABC, abstractmethod
from typing import Dict

import torch
import torch.nn as nn
from esm.model.esm2 import ESM2
from esm.modules import RobertaLMHead
from torch import Tensor

from evo.tokenization import CodonVocab, Vocab
from peint.models.nets.esm2 import ESM2Flash
from peint.models.nets.transformer import (
    FlashMHADecoderBlock,
    FlashMHAEncoderBlock,
    GeometricTimeEmbedder,
)
from peint.models.nets.utils import (
    _create_padding_mask,
    _create_sequence_mask,
    _expand_chain_attr_to_seqlen,
)


class PretrainedEncoder(ABC, nn.Module):
    """Abstract base class for pretrained encoders."""

    def __init__(self, vocab: Vocab, embed_x_per_chain: bool = False):
        super(PretrainedEncoder, self).__init__()
        self.vocab = vocab
        self.embed_x_per_chain = embed_x_per_chain

    @property
    @abstractmethod
    def embed_dim(self) -> int:
        pass

    @abstractmethod
    def get_in_embedding(self) -> nn.Module:
        pass

    @abstractmethod
    def get_out_lm_head(self) -> nn.Module:
        pass

    @abstractmethod
    def forward(self, x: Tensor, x_sizes: Tensor) -> Tensor:
        pass

    @classmethod
    @abstractmethod
    def from_pretrained(cls, *args, **kwargs) -> "PretrainedEncoder":
        pass


class ESMEncoder(PretrainedEncoder):
    """ESM encoder wrapper."""

    def __init__(
        self,
        vocab: Vocab,
        esm_model: ESM2,
        finetune: bool = False,
        embed_x_per_chain: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(vocab=vocab, embed_x_per_chain=embed_x_per_chain, *args, **kwargs)
        self.esm = esm_model
        self.finetune = finetune
        self.esm.eval() if not self.finetune else self.esm.train()
        self.esm.requires_grad_(self.finetune)  # freeze the ESM model

    def embed_dim(self) -> int:
        return self.esm.embed_dim

    def get_in_embedding(self) -> nn.Embedding:
        # embedding layer (optionally from the encoder)
        in_embedding = nn.Embedding(len(self.vocab), self.embed_dim())
        in_embedding.load_state_dict(self.esm.embed_tokens.state_dict())
        in_embedding.requires_grad_(self.finetune)
        return in_embedding

    def get_out_lm_head(self) -> nn.Module:
        # output layer
        lm_head = RobertaLMHead(
            embed_dim=self.embed_dim(),
            output_dim=len(self.vocab),
            weight=self.esm.embed_tokens.weight,
        )
        lm_head.load_state_dict(self.esm.lm_head.state_dict())
        lm_head.requires_grad_(self.finetune)
        return lm_head

    def forward(self, x: Tensor, x_sizes: Tensor) -> Tensor:
        if self.embed_x_per_chain:
            # Multi-sequence processing: compute ESM per chain
            return self.forward_per_chain(x, x_sizes)
        else:
            # Single-sequence processing: process concatenated sequence
            res = self.esm(x, repr_layers=[self.esm.num_layers], need_head_weights=False)
            esm_s = res["representations"][self.esm.num_layers]
            return esm_s

    def forward_per_chain(self, x: Tensor, x_sizes: Tensor):
        """
        Compute ESM embedding separately for each sequence in the batch (multi-sequence style)
        """
        bsz, seq_len = x.shape
        _num_sequences = (x_sizes > 0).sum(dim=1)
        max_num_sequences = _num_sequences.max().item()

        combined_embedding = torch.zeros(
            (bsz, seq_len, (self.esm.embed_dim)), dtype=torch.float, device=x.device
        )

        for seq_idx in range(max_num_sequences):
            # Shape (B, L), True for the residues of `seq_idx`th sequence of each batch element
            seq_mask = _create_sequence_mask(x_sizes, sequence_idx=seq_idx)
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

            # Compute embeddings for the current sequence
            with torch.no_grad():
                output = self.esm(x_seq, repr_layers=[self.esm.num_layers])
                embedding = output["representations"][self.esm.num_layers]

            # Replace nans with zeros (can happen for sequences with less chains)
            embedding = torch.nan_to_num(embedding, nan=0.0)

            # Add the current sequence embedding to the combined embedding
            combined_embedding += embedding * seq_mask.unsqueeze(-1)

        return combined_embedding

    @classmethod
    def from_pretrained(cls, *args, **kwargs) -> "ESMEncoder":
        """Create an ESMEncoder from a pretrained ESM model."""
        import esm

        _esm_model, esm_vocab = esm.pretrained.esm2_t30_150M_UR50D()
        esm_model = ESM2Flash(
            alphabet="ESM-1b",
            num_layers=_esm_model.num_layers,
            embed_dim=_esm_model.embed_dim,
            attention_heads=_esm_model.attention_heads,
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
        return cls(esm_model=esm_model, vocab=evo_vocab, *args, **kwargs)


class ESMCodonEncoder(ESMEncoder):
    """Encoder that concatenates learned codon embeddings with frozen ESM amino acid embeddings."""

    vocab: CodonVocab

    def __init__(
        self,
        vocab: Vocab,
        codon_vocab: CodonVocab,
        codon_embed_dim: int,
        *args,
        **kwargs,
    ):
        super().__init__(vocab=codon_vocab, *args, **kwargs)
        self.aa_vocab = vocab
        self.codon_embed_dim = codon_embed_dim
        self.codon_to_aa_mapping = codon_vocab.translation_tensor_map(aa_vocab=vocab).cuda()
        print(self.codon_to_aa_mapping.device)
        self.codon_embedding = nn.Embedding(len(self.vocab), self.codon_embed_dim)
        nn.init.normal_(self.codon_embedding.weight, mean=0.0, std=0.02)
        self.codon_embedding.requires_grad_(True)

    def embed_dim(self) -> int:
        return self.esm.embed_dim + self.codon_embed_dim

    def get_in_embedding(self) -> nn.Embedding:
        """Hybrid embedding layer that concatenates codon and amino acid embeddings."""

        class HybridEmbedding(nn.Module):
            def __init__(module_self):
                super().__init__()
                module_self.aa_embedding = nn.Embedding(len(self.aa_vocab), self.esm.embed_dim)
                module_self.aa_embedding.load_state_dict(self.esm.embed_tokens.state_dict())
                module_self.aa_embedding.requires_grad_(self.finetune)

            def forward(module_self, x: Tensor) -> Tensor:
                aa_tok_idx = self.codon_to_aa_mapping[x]  # (B, L)
                aa_emb = module_self.aa_embedding(aa_tok_idx)  # (B, L, esm_embed_dim)
                codon_emb = self.codon_embedding(x)  # (B, L, codon_embed_dim)
                return torch.cat([codon_emb, aa_emb], dim=-1)  # (B, L, codon+esm_embed_dim)

        return HybridEmbedding()

    def get_out_lm_head(self) -> nn.Module:
        # output layer trained from scratch
        # create learnable weight for the lm head
        weight = torch.randn(len(self.vocab), self.embed_dim()) * 0.02
        lm_head_weight = nn.Parameter(weight, requires_grad=True)
        lm_head = RobertaLMHead(
            embed_dim=self.embed_dim(),
            output_dim=len(self.vocab),
            weight=lm_head_weight,
        )
        lm_head.requires_grad_(True)  # always finetune the output layer
        return lm_head

    def forward(self, x: Tensor, x_sizes: Tensor) -> Tensor:
        aa_tok_idx = self.codon_to_aa_mapping[x]  # (B, L)
        if self.embed_x_per_chain:
            # Multi-sequence processing: compute ESM per chain
            aa_hx = self.forward_per_chain(aa_tok_idx, x_sizes)
        else:
            # Single-sequence processing: process concatenated sequence
            res = self.esm(aa_tok_idx, repr_layers=[self.esm.num_layers], need_head_weights=False)
            aa_hx = res["representations"][self.esm.num_layers]

        codon_emb = self.codon_embedding(x)  # (B, L, codon_embed_dim)
        joint_hx = torch.cat([codon_emb, aa_hx], dim=-1)  # (B, L, codon+esm_embed_dim)
        return joint_hx


class PEINT(nn.Module):
    """Unified encoder-decoder transformer for protein evolution modeling.

    This model supports both single-sequence (original PEINT) and multi-sequence (PIPET)
    protein evolution modeling through configurable attention mechanisms and data processing.

    The ESM model encodes sequences, and the final hidden representation goes into an
    encoder/decoder stack. The encoder/decoder stack works as a standard transformer
    but with configurable multi-sequence attention support.

    Key Features:
    - Support for both single and multi-sequence inputs
    - Flexible ESM processing (per-chain or concatenated)
    - Unified size-based mask interface

    Args:
        causal_decoder: If True, use causal attention in the decoder (for autoregressive).
                        If False, use bidirectional attention (for masked prediction / discrete diffusion).
    """

    def __init__(
        self,
        enc_model: PretrainedEncoder,
        evo_vocab: Vocab,
        embed_dim: int,
        num_heads: int,
        num_chains: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        max_len=1022,
        dropout_p: float = 0.0,
        use_chain_embedding: bool = True,
        use_attention_bias: bool = True,
        causal_decoder: bool = True,
    ):
        super(PEINT, self).__init__()

        self.vocab = evo_vocab
        self.max_len = max_len
        self.enc_model = enc_model
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_p = dropout_p
        self.use_bias = use_attention_bias
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers

        assert self.embed_dim == enc_model.embed_dim()
        assert num_encoder_layers >= num_decoder_layers

        # Loss function
        self.criterion = nn.CrossEntropyLoss(reduction="mean", ignore_index=self.vocab.pad_idx)

        # Always use multi-sequence blocks (they handle single sequences as special case)
        self.enc_layers = nn.ModuleList(
            [
                FlashMHAEncoderBlock(
                    embed_dim=self.embed_dim,
                    ffn_embed_dim=4 * self.embed_dim,
                    attention_heads=self.num_heads,
                    add_bias_kv=False,
                    dropout_p=self.dropout_p,
                    use_bias=self.use_bias,
                    layer_idx=l,
                )
                for l in range(num_encoder_layers)
            ]
        )
        self.dec_layers = nn.ModuleList(
            [
                FlashMHADecoderBlock(
                    embed_dim=self.embed_dim,
                    ffn_embed_dim=4 * self.embed_dim,
                    attention_heads=self.num_heads,
                    add_bias_kv=False,
                    dropout_p=self.dropout_p,
                    use_bias=self.use_bias,
                    layer_idx=l,
                    causal=causal_decoder,
                )
                for l in range(num_decoder_layers)
            ]
        )

        # time embedding
        self.time_embedding = GeometricTimeEmbedder(
            frequency_embedding_size=embed_dim, start=1e-5, stop=0.25
        )

        # chain embedding (0 is padding idx)
        self.chain_embedding = (
            nn.Embedding(num_chains + 1, embed_dim, padding_idx=0) if use_chain_embedding else None
        )

        # get input and output layers from the encoder
        self.in_embedding = self.enc_model.get_in_embedding()
        self.out_lm_head = self.enc_model.get_out_lm_head()

    def forward(
        self,
        x: Tensor,
        y: Tensor,
        t: Tensor,
        x_sizes: Tensor,
        y_sizes: Tensor,
        chain_ids: Tensor = None,
    ) -> Dict[str, Tensor]:
        """
        Unified forward pass using standardized size-based interface.

        Args:
            x: Encoder input sequences (batch_size, seq_len)
            y: Decoder input sequences (batch_size, seq_len)
            t: Time values (batch_size, num_chains)
            x_sizes: Sequence sizes for encoder (batch_size, seq_len)
            y_sizes: Sequence sizes for decoder (batch_size, seq_len)
            chain_ids: Chain IDs for each sequence in the batch (batch_size, num_chains)

        Returns:
            Dict with 'enc_logits' and 'dec_logits' keys
        """
        # Embed decoder input with time
        h_y = self.in_embedding(y)

        # Position-specific time embedding using distances
        distances_expanded = _expand_chain_attr_to_seqlen(
            chain_attr=t, sizes=y_sizes, pad_value=0.0
        )
        ht = self.time_embedding(distances_expanded)
        h_y = h_y + ht

        # Create attention masks from sizes (1 for data, 0 for pad)
        x_attn_mask = _create_padding_mask(x_sizes)
        y_attn_mask = _create_padding_mask(y_sizes)

        # Get pretrained encoder representations
        h_x = self.enc_model.forward(x=x, x_sizes=x_sizes)

        # Add chain embeddings to both h_x and h_y if available
        if self.chain_embedding is not None and chain_ids is not None:
            chain_attr_expanded_x = _expand_chain_attr_to_seqlen(
                chain_attr=chain_ids, sizes=x_sizes, pad_value=0
            )
            chain_attr_expanded_y = _expand_chain_attr_to_seqlen(
                chain_attr=chain_ids, sizes=y_sizes, pad_value=0
            )
            h_x = h_x + self.chain_embedding(chain_attr_expanded_x.long())
            h_y = h_y + self.chain_embedding(chain_attr_expanded_y.long())

        # Encoder-decoder forward pass
        for i, enc_layer in enumerate(self.enc_layers):
            # Encoder layer
            h_x = enc_layer(x=h_x, x_attn_mask=x_attn_mask)

            # Decoder layers (with encoder-decoder alignment)
            if self.num_decoder_layers - self.num_encoder_layers + i >= 0:
                idx = self.num_decoder_layers - self.num_encoder_layers + i
                dec_layer = self.dec_layers[idx]
                # Decoder layer (args: x means main input, y is cross attended input)
                h_y = dec_layer(x=h_y, x_attn_mas=y_attn_mask, y=h_x, y_attn_mask=x_attn_mask)

        # Generate logits
        x_logits = self.out_lm_head(h_x)
        y_logits = self.out_lm_head(h_y)

        if torch.any(torch.isnan(x_logits)) or torch.any(torch.isnan(y_logits)):
            raise ValueError("NaN detected in logits")

        # Return dictionary format
        return dict(enc_logits=x_logits, dec_logits=y_logits)
