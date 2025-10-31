from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from evo.tokenization import Vocab
from peint.models.frameworks.nde import NeuralGeodesicFlows
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
    """
    Transformer-based encoder for protein sequences embeddings (L,d_enc) to latent vectors (h)
    """

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
    """
    Transformer-based decoder for protein sequences conditioned on latent vectors
    """

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


class GeodesicMetric(nn.Module):
    """
    Learns an hxh positive definite matrix for computing the geodesic metric in the latent space
    """

    def __init__(self, M_dim: int, hidden_dims: List[int], activation: str = "tanh"):
        super().__init__()
        self.M_dim = M_dim
        self.activation = activation

        # Build layers: input -> hidden layers -> lower triangular output
        layer_sizes = [M_dim] + hidden_dims + [M_dim * (M_dim + 1) // 2]
        self.layers = nn.ModuleList(
            [nn.Linear(layer_sizes[i], layer_sizes[i + 1]) for i in range(len(layer_sizes) - 1)]
        )

    def activation_fn(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "tanh":
            return torch.tanh(x)
        elif self.activation == "relu":
            return torch.relu(x)
        else:
            raise ValueError(f"Unsupported activation function: {self.activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs, _ = x.shape

        # Hidden layers with tanh activation
        for layer in self.layers[:-1]:
            x = self.activation_fn(layer(x))

        # Output layer with tanh
        learnt_components = self.activation_fn(self.layers[-1](x))  # (bs, n_tril)

        # Build lower triangular matrix L using scatter (out-of-place)
        tril_indices = torch.tril_indices(self.M_dim, self.M_dim, device=x.device)
        flat_indices = (tril_indices[0] * self.M_dim + tril_indices[1]).unsqueeze(0).expand(bs, -1)

        L_flat = torch.zeros(bs, self.M_dim * self.M_dim, device=x.device, dtype=x.dtype)
        L_flat = L_flat.scatter(1, flat_indices, learnt_components)
        L = L_flat.reshape(bs, self.M_dim, self.M_dim)

        # Apply softplus to diagonal (out-of-place)
        diag_mask = torch.eye(self.M_dim, device=x.device, dtype=torch.bool)
        L = torch.where(diag_mask, nn.functional.softplus(L), L)

        # Compute g = I + LL^T (guaranteed symmetric positive definite)
        I = torch.eye(self.M_dim, device=x.device, dtype=x.dtype).unsqueeze(0)
        g = I + torch.bmm(L, L.transpose(1, 2))

        return g


class LEPT(NeuralGeodesicFlows):
    def __init__(
        self,
        esm_encoder: ESMEncoder,
        vocab: Vocab,
        latent_dim: int = 128,
        encoder_num_adaptive_layers: int = 3,
        decoder_num_layers: int = 6,
        decoder_num_heads: int = 8,
        num_steps: int = 100,
        max_len: int = 1024,
        is_vae: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(latent_dim=latent_dim, *args, **kwargs)
        self.vocab = vocab
        self.latent_dim = latent_dim
        self.esm_encoder = esm_encoder
        self.num_steps = num_steps
        self.is_vae = is_vae

        in_embedding = esm_encoder.get_in_embedding()
        out_lm_head = esm_encoder.get_out_lm_head()

        self.encoder = SequenceEncoder(
            esm_encoder.embed_dim,
            latent_dim,
            encoder_num_adaptive_layers,
        )
        self.decoder = SequenceDecoder(
            embedding=in_embedding,
            lm_head=out_lm_head,
            latent_dim=latent_dim,
            embed_dim=esm_encoder.embed_dim,
            num_layers=decoder_num_layers,
            num_heads=decoder_num_heads,
            max_len=max_len,
        )
        self.metric = GeodesicMetric(
            M_dim=latent_dim // 2,
            hidden_dims=[256, 256],
        )

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: Tensor, x_sizes: Tensor) -> dict[str, Tensor]:
        attn_mask = (x != self.vocab.pad_idx).long()

        h = self.esm_encoder(x, x_sizes)
        mu, logvar = self.encoder(h, attn_mask)

        if self.is_vae:
            z = self.reparameterize(mu, logvar)
        else:
            z = mu

        x_dec = x[:, :-1]
        attn_mask_dec = attn_mask[:, :-1]
        logits = self.decoder(x_dec, z, attn_mask_dec)

        return {"logits": logits, "mu": mu, "logvar": logvar, "z": z}

    def recon_loss(self, x, y, x_sizes, y_sizes, calc_acc=False, **kwargs):
        x_outputs = self.forward(x, x_sizes)
        y_outputs = self.forward(y, y_sizes)

        # combine outputs for both x and y (treat them identically)
        outputs = {k: torch.cat([x_outputs[k], y_outputs[k]], dim=0) for k in x_outputs}
        logits, mu, logvar = outputs["logits"], outputs["mu"], outputs["logvar"]

        # combine the targets for both x and y as well
        target = torch.cat([x[:, 1:], y[:, 1:]], dim=0)

        recon_loss = F.cross_entropy(
            logits.transpose(1, 2),
            target,
            ignore_index=self.vocab.pad_idx,
            reduction="mean",
        )
        loss = recon_loss
        loss_info = dict(recon_loss=recon_loss)

        if self.is_vae:
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            beta = kwargs.get("beta", 1.0)
            loss_info["kl_loss"] = kl_loss
            loss += beta * kl_loss

        if calc_acc:
            padding_mask = target != self.vocab.pad_idx
            acc = (logits.argmax(-1)[padding_mask] == target[padding_mask]).float().mean().item()
            loss_info["recon_acc"] = acc

        loss_info["loss"] = loss
        return loss_info

    def trans_loss(self, x, y, x_sizes, y_sizes, t, **kwargs):
        # Encode input sequences to latent space
        Z_x = self.encode(x, x_sizes)
        Z_y = self.encode(y, y_sizes)

        # Apply the geodesic exp map to simulate Z_x over time t
        Z_y_hat = self.exp(Z_x, t, num_steps=self.num_steps)

        # Calculate cross-entropy loss between decoded sequences from Z_y_hat and target y
        y_attn_mask = (y != self.vocab.pad_idx).long()
        logits = self.decoder(y[:, :-1], Z_y_hat, y_attn_mask[:, :-1])

        trans_lat_loss = F.mse_loss(  # latent space transition loss
            input=Z_y_hat,
            target=Z_y,
            reduction="mean",
        )
        trans_dec_loss = F.cross_entropy(  # decoded space transition loss
            logits.transpose(1, 2),
            y[:, 1:],
            ignore_index=self.vocab.pad_idx,
            reduction="mean",
        )
        loss = trans_lat_loss + trans_dec_loss
        return dict(loss=loss, trans_lat_loss=trans_lat_loss, trans_dec_loss=trans_dec_loss)

    def loss(self, x, y, x_sizes, y_sizes, t, beta, **kwargs):
        recon_loss_dict = self.recon_loss(
            x=x,
            y=y,
            x_sizes=x_sizes,
            y_sizes=y_sizes,
            beta=beta,
            **kwargs,
        )
        trans_loss_dict = self.trans_loss(
            x=x,
            y=y,
            x_sizes=x_sizes,
            y_sizes=y_sizes,
            t=t,
            **kwargs,
        )
        loss = recon_loss_dict["loss"] + trans_loss_dict["loss"]
        loss_info = {**recon_loss_dict, **trans_loss_dict, "loss": loss}
        return loss_info

    @torch.no_grad()
    def encode(self, x: Tensor, x_sizes: Tensor) -> Tensor:
        attn_mask = (x != self.vocab.pad_idx).long()
        h = self.esm_encoder(x, x_sizes)
        mu, _ = self.encoder(h, attn_mask)
        return mu

    @torch.no_grad()
    def evolve(self, z: Tensor, t: Tensor, num_steps: int) -> Tensor:
        return self.exp(z, t, num_steps=num_steps)

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
