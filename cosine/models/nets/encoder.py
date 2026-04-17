from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from esm.model.esm2 import ESM2
from esm.modules import RobertaLMHead
from torch import Tensor

from evo.tokenization import Vocab
from cosine.models.nets.esm2 import ESM2Flash
from cosine.models.nets.utils import _create_sequence_mask


class PretrainedEncoder(ABC, nn.Module):
    """Abstract base class for pretrained encoders."""

    def __init__(self, vocab: Vocab, embed_x_per_chain: bool = False):
        super().__init__()
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

    @property
    def embed_dim(self) -> int:
        return self.esm.embed_dim

    def get_in_embedding(self) -> nn.Module:
        # embedding layer (optionally from the encoder)
        in_embedding = nn.Embedding(len(self.vocab), self.embed_dim)
        in_embedding.load_state_dict(self.esm.embed_tokens.state_dict())
        in_embedding.requires_grad_(self.finetune)
        return in_embedding

    def get_out_lm_head(self) -> nn.Module:
        # output layer
        lm_head = RobertaLMHead(
            embed_dim=self.embed_dim,
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
        max_num_sequences = _num_sequences.max().int().item()

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

        _esm_model, esm_vocab = esm.pretrained.esm2_t6_8M_UR50D()
        # _esm_model, esm_vocab = esm.pretrained.esm2_t30_150M_UR50D()
        # _esm_model, esm_vocab = esm.pretrained.esm2_t33_650M_UR50D()

        esm_model = ESM2Flash(
            alphabet="ESM-1b",
            num_layers=_esm_model.num_layers,
            embed_dim=_esm_model.embed_dim,
            attention_heads=_esm_model.attention_heads,
            token_dropout=_esm_model.token_dropout,
            dropout_p=kwargs.get("dropout_p", 0.0),
        )

        ckpt_path = kwargs.pop("ckpt_path", None)
        load_pretrained = kwargs.pop("load_pretrained", True)

        if ckpt_path is not None:
            state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
            esm_model.load_state_dict(state_dict=state_dict, strict=False)
        elif load_pretrained:
            print("Loading pretrained ESM model from existing weights")
            esm_model.load_state_dict(_esm_model.state_dict(), strict=False)
            del _esm_model
        else:
            print("Initializing ESM model from scratch")
            del _esm_model  # random initialization

        evo_vocab = Vocab.from_esm_alphabet(esm_vocab)
        return cls(esm_model=esm_model, vocab=evo_vocab, *args, **kwargs)
