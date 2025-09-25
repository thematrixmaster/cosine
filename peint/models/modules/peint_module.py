from functools import partial
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.optimization import get_polynomial_decay_schedule_with_warmup

from peint.metrics.api import Metric
from peint.models.modules.plmr_module import PLMRLitModule
from peint.models.nets.peint import PEINT


class PEINTModule(PLMRLitModule):
    """Unified Lightning module for training PEINT models"""

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
        metrics: Dict[str, Metric] = {},
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
        self.metrics = metrics

    def forward(self, *args, **kwargs):
        """Forward method delegates to unified network interface"""
        return self.net(*args, **kwargs)

    def model_step(self, batch):
        """Simplified model step using unified network interface"""
        # New unified format: [x_src, x_tgt, y_src, y_tgt, ts, x_sizes, y_sizes]
        x_src, x_tgt, y_src, y_tgt, ts, x_sizes, y_sizes = batch

        # Use unified network interface with size-based masks
        outputs = self.net(x_src, y_src, ts, x_sizes=x_sizes, y_sizes=y_sizes)

        # Handle outputs from network
        x_logits, y_logits = outputs["enc_logits"], outputs["dec_logits"]

        if isinstance(self.net.criterion, nn.CrossEntropyLoss):
            # cross entropy loss expects the logits to be in the second to last dimension
            x_logits = x_logits.transpose(-1, -2)
            y_logits = y_logits.transpose(-1, -2)

        mlm_loss = self.net.criterion(x_logits, x_tgt)
        tlm_loss = self.net.criterion(y_logits, y_tgt)

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

    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()
        self.validation_step_outputs = [[] for _ in self.trainer.val_dataloaders]

    def validation_step(self, batch, batch_idx: int, dataloader_idx: int):
        """Simplified validation step using unified network interface"""
        # Handle different data formats
        if dataloader_idx == 0:
            # Standard training format: [x_src, x_tgt, y_src, y_tgt, ts, x_sizes, y_sizes]
            x_src, x_tgt, y_src, y_tgt, ts, x_sizes, y_sizes = batch
        else:
            # DMS format: [x_src, y_src, y_tgt, muts, ts, x_sizes, y_sizes]
            x_src, y_src, y_tgt, muts, ts, x_sizes, y_sizes = batch
            x_tgt = None

        with torch.no_grad():
            outputs = self.net(x_src, y_src, ts, x_sizes=x_sizes, y_sizes=y_sizes)

        # Handle outputs from network
        x_logits, y_logits = outputs["enc_logits"], outputs["dec_logits"]
        loss_info = {}

        # Calculate MLM loss only if we have encoder targets (dataloader_idx == 0)
        if dataloader_idx == 0 and x_tgt is not None:
            mlm_loss = F.cross_entropy(
                x_logits.transpose(-1, -2),
                x_tgt,
                ignore_index=self.net.vocab.pad_idx,
                reduction="mean",
            )
            mlm_ppl = torch.exp(mlm_loss.detach())
            loss_info["mlm_loss"] = mlm_loss
            loss_info["mlm_ppl"] = mlm_ppl

        # Keep unreduced to get per-site time likelihood
        clm_loss = F.cross_entropy(
            y_logits.transpose(-1, -2),
            y_tgt,
            ignore_index=self.net.vocab.pad_idx,
            reduction="none",
        ).detach()

        if dataloader_idx > 0:  # DMS dataset
            padding_mask = y_tgt != self.net.vocab.pad_idx
            nll = (clm_loss * padding_mask).mean(-1)
            ppls = torch.exp(nll).detach().cpu().numpy()
            self.validation_step_outputs[dataloader_idx].append((ppls, muts))
            return ppls, muts

        # Standard validation (dataloader_idx == 0)
        padding_mask = y_tgt != self.net.vocab.pad_idx
        clm_loss_mean = clm_loss[padding_mask].mean()
        clm_ppl = torch.exp(clm_loss_mean)

        if "mlm_loss" in loss_info:
            loss = loss_info["mlm_loss"] + clm_loss_mean
        else:
            loss = clm_loss_mean

        acc = (y_logits.argmax(-1)[padding_mask] == y_tgt[padding_mask]).float().mean().item()

        loss_info.update(
            {
                "clm_loss": clm_loss_mean,
                "clm_ppl": clm_ppl,
                "acc": acc,
            }
        )

        self.val_loss(clm_loss_mean)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

        for k, v in loss_info.items():
            self.log(f"val/{k}", v, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return {"loss": loss, **loss_info}

    def on_validation_epoch_end(self, *args, **kwargs) -> None:
        super().on_validation_epoch_end(*args, **kwargs)
        if (
            not self.trainer.sanity_checking
            and (self.current_epoch % self.hparams.val_metrics_every_n_epoch) == 0
        ):
            for _, metric in self.metrics.items():
                _metrics = metric.compute(self.validation_step_outputs)
                _metrics = metric.aggregate(_metrics)
                for key, value in _metrics.items():
                    self.log(f"val/{metric.name}/{key}", value, sync_dist=True, prog_bar=True)

    def predict_step(self, batch, batch_idx):
        raise NotImplementedError("Predict step not implemented yet.")

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
