from functools import partial
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.optimization import get_polynomial_decay_schedule_with_warmup

from peint.models.modules.plmr_module import PLMRLitModule
from peint.models.nets.peint import PEINT, PIPET
from peint.models.nets.utils import _create_sequence_mask, _expand_distances_to_seqlen


class PEINTModule(PLMRLitModule):
    """Lightning module for training a T5 PEINT model"""

    def __init__(
        self,
        net: PEINT,
        weight_decay: float = 0.01,
        optimizer: torch.optim.Optimizer = partial(torch.optim.AdamW, lr=1e-4),
        scheduler: torch.optim.lr_scheduler = partial(
            get_polynomial_decay_schedule_with_warmup,
            num_warmup_steps=10000,
            num_training_steps=100000,
            power=2.0,
        ),
        compile: bool = False,
        val_metrics_every_n_epoch: int = 1,
        ignore: List[str] = [],
        *args,
        **kwargs,
    ):
        super().__init__(
            net=net,
            optimizer=optimizer,
            scheduler=scheduler,
            compile=compile,
            ignore=ignore,
            *args,
            **kwargs,
        )

    def forward(self, x, y, t, x_pad_mask, y_pad_mask):
        return self.net(x, y, t, x_pad_mask, y_pad_mask)

    def model_step(self, batch):
        # assumes y & y_targets are already shifted for auto-regressive decoding
        [x, x_targets, y, y_targets, t, x_pad_mask, y_pad_mask] = batch

        logits = self(x, y, t, x_pad_mask, y_pad_mask)

        if isinstance(self.net.criterion, nn.CrossEntropyLoss):
            # cross entropy loss expects the logits to be in the second to last dimension
            logits = [l.transpose(-1, -2) for l in logits]

        x_logits, y_logits = logits

        mlm_loss = self.net.criterion(x_logits, x_targets)
        tlm_loss = self.net.criterion(y_logits, y_targets)

        mlm_ppl = torch.exp(mlm_loss.detach())
        tlm_ppl = torch.exp(tlm_loss.detach())

        loss = mlm_loss + tlm_loss

        metrics = {
            "mlm_loss": mlm_loss,
            "tlm_loss": tlm_loss,
            "mlm_ppl": mlm_ppl,
            "tlm_ppl": tlm_ppl,
        }
        return loss, metrics

    def validation_step(self, batch, batch_idx):
        [x, x_targets, y, y_targets, t, x_pad_mask, y_pad_mask] = batch

        yt_mask = y_targets != self.net.vocab.pad_idx  # actual values

        times = t.expand_as(y_targets)  # expand time to match y_targets
        tbins = times[yt_mask]  # get time bins for actual values

        with torch.no_grad():
            x_logits, y_logits = self(x, y, t, x_pad_mask, y_pad_mask)

        y_loss = F.cross_entropy(
            y_logits.transpose(-1, -2),
            y_targets,
            ignore_index=self.net.vocab.pad_idx,
            reduction="none",
        )  # keep unreduced to get per-site time likelihood

        mlm_loss = F.cross_entropy(
            x_logits.transpose(-1, -2),
            x_targets,
            ignore_index=self.net.vocab.pad_idx,
            reduction="mean",
        )  # mlm doesnt' care about time

        mlm_ppl = torch.exp(mlm_loss.detach())

        mask_loss = -1 * y_loss[yt_mask]  # log likelihoods (bs, seq_len)

        ppl_per_bin = {
            b.item(): mask_loss[tbins == b].cpu().numpy() for b in t
        }  # gets the actual bin idx

        acc = (y_logits.argmax(-1)[yt_mask] == y_targets[yt_mask]).float().mean().detach()

        loss = y_loss[yt_mask].mean()
        loss_info = {
            "ppl": torch.exp(y_loss[yt_mask].mean()),
            "acc": acc,
            "mlm_loss": mlm_loss,
            "mlm_ppl": mlm_ppl,
        }

        # update and log validation loss
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

        # log additional loss info
        for k, v in loss_info.items():
            self.log(f"val/{k}", v, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return {"loss": loss, **loss_info}, ppl_per_bin

    def predict_step(self, batch, batch_idx):
        [x, t] = batch
        y_decoded = self.net.generate(
            x=x,
            t=t,
            max_decode_steps=self.hparams.get("max_decode_steps", 1022),
            device=self.device,
            temperature=self.hparams.get("temperature", 1.0),
            p=self.hparams.get("p", 1.0),
        )
        return y_decoded

    def configure_optimizers(self):
        # multiple param groups for the encoder and decoder
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": self.hparams.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]

        optimizer = self.hparams.optimizer(params=optim_groups)
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "train/loss",
                    "interval": "step",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    def state_dict(self, *args, **kwargs):
        # Remove ESM2 specific keys from the state dict, if frozen
        state_dict = super().state_dict(*args, **kwargs)
        if not self.net.finetune_esm:
            keys_to_remove = [key for key in state_dict.keys() if "net.esm" in key]
            for key in keys_to_remove:
                del state_dict[key]
        return state_dict


class PIPETModule(PEINTModule):
    def __init__(
        self,
        net: PIPET,
        weight_decay: float = 0.01,
        optimizer: torch.optim.Optimizer = partial(torch.optim.AdamW, lr=1e-4),
        scheduler: torch.optim.lr_scheduler = partial(
            get_polynomial_decay_schedule_with_warmup,
            num_warmup_steps=10000,
            num_training_steps=100000,
            power=2.0,
        ),
        compile: bool = False,
        val_metrics_every_n_epoch: int = 1,
        ignore: List[str] = [],
        *args,
        **kwargs,
    ):
        super().__init__(
            net=net,
            optimizer=optimizer,
            scheduler=scheduler,
            compile=compile,
            ignore=ignore,
            *args,
            **kwargs,
        )

    def forward(self, enc_in, dec_in, enc_attn_mask, dec_attn_mask, distances):
        return self.net(enc_in, dec_in, enc_attn_mask, dec_attn_mask, distances)

    def model_step(self, batch):
        [
            enc_inputs,
            enc_targets,
            dec_inputs,
            dec_targets,
            distances_tensor,
            attn_mask_enc_lengths,
            attn_mask_dec_lengths,
        ] = batch

        outputs = self(
            enc_inputs,
            dec_inputs,
            attn_mask_enc_lengths,
            attn_mask_dec_lengths,
            distances_tensor,
        )
        enc_logits, dec_logits = outputs["enc_logits"], outputs["dec_logits"]

        if isinstance(self.net.criterion, nn.CrossEntropyLoss):
            # Cross entropy loss expects the logits to be in the second to last dimension
            enc_logits = enc_logits.transpose(-1, -2)
            dec_logits = dec_logits.transpose(-1, -2)

        mlm_loss = self.net.criterion(enc_logits, enc_targets)
        tlm_loss = self.net.criterion(dec_logits, dec_targets)

        mlm_ppl = torch.exp(mlm_loss.detach())
        tlm_ppl = torch.exp(tlm_loss.detach())

        loss = mlm_loss + tlm_loss

        metrics = {
            "mlm_loss": mlm_loss,
            "tlm_loss": tlm_loss,
            "mlm_ppl": mlm_ppl,
            "tlm_ppl": tlm_ppl,
        }
        return loss, metrics

    def validation_step(self, batch, batch_idx):
        [
            enc_inputs,
            enc_targets,
            dec_inputs,
            dec_targets,
            distances_tensor,
            attn_mask_enc_lengths,
            attn_mask_dec_lengths,
        ] = batch

        with torch.no_grad():
            outputs = self(
                enc_inputs,
                dec_inputs,
                attn_mask_enc_lengths,
                attn_mask_dec_lengths,
                distances_tensor,
            )
        enc_logits, dec_logits = outputs["enc_logits"], outputs["dec_logits"]

        mlm_loss = F.cross_entropy(
            enc_logits.transpose(-1, -2),
            enc_targets,
            ignore_index=self.net.vocab.pad_idx,
            reduction="mean",
        )
        mlm_ppl = torch.exp(mlm_loss.detach())

        # Keep unreduced to get per-site time likelihood
        clm_loss = F.cross_entropy(
            dec_logits.transpose(-1, -2),
            dec_targets,
            ignore_index=self.net.vocab.pad_idx,
            reduction="none",
        ).detach()

        # Obtain the per-site likelihood for x2 and y2 separately
        # Generate mask for x2 and y2 in the decoder targets
        x_seq_mask = _create_sequence_mask(attn_mask_dec_lengths, sequence_idx=0)
        y_seq_mask = _create_sequence_mask(attn_mask_dec_lengths, sequence_idx=1)
        # Expand the distance to be per position, shape (B, L)
        distances_seqlen = _expand_distances_to_seqlen(distances_tensor, attn_mask_dec_lengths)

        # Shape (total num tokens of x in the batch,)
        x_dist = distances_seqlen[x_seq_mask]
        x_ll = -clm_loss[x_seq_mask]
        # Dict: distance --> per-site log likelihood of x
        x_ll_per_dist = {
            d.item(): x_ll[x_dist == d].cpu().numpy() for d in distances_tensor[:, 0].unique()
        }
        # Shape (total num tokens of y in the batch,)
        y_dist = distances_seqlen[y_seq_mask]
        y_ll = -clm_loss[y_seq_mask]
        y_ll_per_dist = {
            d.item(): y_ll[y_dist == d].cpu().numpy() for d in distances_tensor[:, 1].unique()
        }
        ll_per_dist = (x_ll_per_dist, y_ll_per_dist)

        # Average decoder metrics
        padding_mask = dec_targets != self.net.vocab.pad_idx
        clm_loss_mean = clm_loss[padding_mask].mean()
        clm_ppl = torch.exp(clm_loss_mean)
        loss = mlm_loss + clm_loss_mean
        acc = (
            (dec_logits.argmax(-1)[padding_mask] == dec_targets[padding_mask]).float().mean().item()
        )
        loss_info = {
            "clm_loss": clm_loss_mean,
            "clm_ppl": clm_ppl,
            "acc": acc,
            "mlm_loss": mlm_loss,
            "mlm_ppl": mlm_ppl,
        }

        # update and log validation loss
        self.val_loss(clm_loss_mean)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

        # log additional loss info
        for k, v in loss_info.items():
            self.log(f"val/{k}", v, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return {"loss": loss, **loss_info}, ll_per_dist

    def predict_step(self, batch, batch_idx):
        raise NotImplementedError("PIPETModule does not implement predict_step")

