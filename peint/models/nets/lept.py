import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from evo.tokenization import Vocab
from peint.models.nets.peint import ESMEncoder


class AttentionPooling(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)

    def forward(self, x: Tensor, attn_mask: Tensor) -> Tensor:
        bsz = x.size(0)
        query = self.query.expand(bsz, -1, -1)
        pooled, _ = self.attn(query, x, x, key_padding_mask=~attn_mask.bool())
        return pooled.squeeze(1)


class SequenceEncoder(nn.Module):
    def __init__(self, esm_embed_dim: int, latent_dim: int, num_adaptive_layers: int = 3):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=esm_embed_dim,
            nhead=8,
            dim_feedforward=4 * esm_embed_dim,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.adaptive_layers = nn.TransformerEncoder(encoder_layer, num_adaptive_layers)
        self.pooling = AttentionPooling(esm_embed_dim)
        self.fc_mu = nn.Linear(esm_embed_dim, latent_dim)
        self.fc_logvar = nn.Linear(esm_embed_dim, latent_dim)

    def forward(self, h: Tensor, attn_mask: Tensor) -> tuple[Tensor, Tensor]:
        src_key_padding_mask = ~attn_mask.bool()
        h = self.adaptive_layers(h, src_key_padding_mask=src_key_padding_mask)
        pooled = self.pooling(h, attn_mask)
        mu = self.fc_mu(pooled)
        logvar = self.fc_logvar(pooled)
        return mu, logvar


class SequenceDecoder(nn.Module):
    def __init__(
        self,
        embedding: nn.Module,
        lm_head: nn.Module,
        latent_dim: int,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        max_len: int = 1024,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.embedding = embedding
        self.lm_head = lm_head
        self.latent_proj = nn.Linear(latent_dim, embed_dim)
        self.pos_embedding = nn.Embedding(max_len, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=4 * embed_dim,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, x: Tensor, z: Tensor, attn_mask: Tensor) -> Tensor:
        bsz, seq_len = x.shape

        x_emb = self.embedding(x)
        pos = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(bsz, -1)
        x_emb = x_emb + self.pos_embedding(pos)

        z_proj = self.latent_proj(z).unsqueeze(1).expand(-1, seq_len, -1)
        x_emb = x_emb + z_proj

        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=x.device)
        src_key_padding_mask = ~attn_mask.bool()

        h = self.decoder(
            src=x_emb,
            mask=causal_mask,
            src_key_padding_mask=src_key_padding_mask,
        )

        logits = self.lm_head(h)
        return logits


class ProteinVAE(nn.Module):
    def __init__(
        self,
        esm_encoder: ESMEncoder,
        vocab: Vocab,
        latent_dim: int = 128,
        encoder_num_adaptive_layers: int = 3,
        decoder_num_layers: int = 6,
        decoder_num_heads: int = 8,
        max_len: int = 1024,
    ):
        super().__init__()
        self.vocab = vocab
        self.latent_dim = latent_dim
        self.esm_encoder = esm_encoder

        self.encoder = SequenceEncoder(
            esm_encoder.embed_dim(), latent_dim, encoder_num_adaptive_layers
        )

        in_embedding = esm_encoder.get_in_embedding()
        out_lm_head = esm_encoder.get_out_lm_head()

        self.decoder = SequenceDecoder(
            embedding=in_embedding,
            lm_head=out_lm_head,
            latent_dim=latent_dim,
            embed_dim=esm_encoder.embed_dim(),
            num_layers=decoder_num_layers,
            num_heads=decoder_num_heads,
            max_len=max_len,
        )

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: Tensor, x_sizes: Tensor) -> dict[str, Tensor]:
        attn_mask = (x != self.vocab.pad_idx).long()

        h = self.esm_encoder(x, x_sizes)
        mu, logvar = self.encoder(h, attn_mask)
        z = self.reparameterize(mu, logvar)

        x_dec = x[:, :-1]
        attn_mask_dec = attn_mask[:, :-1]
        logits = self.decoder(x_dec, z, attn_mask_dec)

        return {"logits": logits, "mu": mu, "logvar": logvar, "z": z}

    def loss(self, x: Tensor, x_sizes: Tensor, beta: float = 1.0) -> dict[str, Tensor]:
        outputs = self.forward(x, x_sizes)
        logits = outputs["logits"]
        mu = outputs["mu"]
        logvar = outputs["logvar"]

        target = x[:, 1:]

        recon_loss = F.cross_entropy(
            logits.transpose(1, 2),
            target,
            ignore_index=self.vocab.pad_idx,
            reduction="mean",
        )

        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        loss = recon_loss + beta * kl_loss

        return {
            "loss": loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
        }

    @torch.no_grad()
    def encode(self, x: Tensor, x_sizes: Tensor) -> Tensor:
        attn_mask = (x != self.vocab.pad_idx).long()
        h = self.esm_encoder(x, x_sizes)
        mu, _ = self.encoder(h, attn_mask)
        return mu

    @torch.no_grad()
    def decode(self, z: Tensor, max_len: int, temperature: float = 1.0) -> Tensor:
        bsz = z.size(0)
        device = z.device

        x = torch.full((bsz, 1), self.vocab.bos_idx, dtype=torch.long, device=device)
        eos_reached = torch.zeros(bsz, dtype=torch.bool, device=device)

        for _ in range(max_len - 1):
            attn_mask = torch.ones_like(x)
            logits = self.decoder(x, z, attn_mask)

            next_logits = logits[:, -1, :] / temperature
            next_token = torch.argmax(next_logits, dim=-1, keepdim=True)

            x = torch.cat([x, next_token], dim=1)

            is_eos_token = next_token.squeeze(1) == self.vocab.eos_idx
            eos_reached = eos_reached | is_eos_token

            if eos_reached.all():
                break

        return x
