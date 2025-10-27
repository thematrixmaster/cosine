from functools import partial
from typing import List

import torch
import torch.nn.functional as F
from transformers.optimization import get_polynomial_decay_schedule_with_warmup

from peint.models.modules.plmr_module import PLMRLitModule
from peint.models.nets.lept import ProteinVAE


class ProteinVAEModule(PLMRLitModule):
    def __init__(
        self,
        net: ProteinVAE,
        weight_decay: float = 0.01,
        optimizer: torch.optim.Optimizer = partial(torch.optim.AdamW, lr=1e-4),
        scheduler: torch.optim.lr_scheduler = partial(
            get_polynomial_decay_schedule_with_warmup,
            num_warmup_steps=10000,
            num_training_steps=100000,
            power=2.0,
        ),
        kl_anneal_steps: int = 10000,
        beta_max: float = 1.0,
        compile: bool = False,
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
        self.kl_anneal_steps = kl_anneal_steps
        self.beta_max = beta_max

    def get_beta(self) -> float:
        if self.kl_anneal_steps == 0:
            return self.beta_max
        step = self.global_step
        return min(self.beta_max, step / self.kl_anneal_steps)

    def forward(self, *args, **kwargs):
        return self.net(*args, **kwargs)

    def model_step(self, batch):
        x, y, t, x_sizes, y_sizes = batch

        beta = self.get_beta()

        loss_dict = self.net.loss(x=x, x_sizes=x_sizes, beta=beta)

        loss = loss_dict["loss"]
        recon_loss = loss_dict["recon_loss"]
        kl_loss = loss_dict["kl_loss"]

        recon_ppl = torch.exp(recon_loss.detach())

        metrics = {
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "recon_ppl": recon_ppl,
            "beta": beta,
        }
        return loss, metrics

    def validation_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        x, y, t, x_sizes, y_sizes = batch

        beta = self.beta_max

        with torch.no_grad():
            outputs = self.net(x=x, x_sizes=x_sizes)

        logits = outputs["logits"]
        mu = outputs["mu"]
        logvar = outputs["logvar"]

        target = x[:, 1:]

        recon_loss = F.cross_entropy(
            logits.transpose(1, 2),
            target,
            ignore_index=self.net.vocab.pad_idx,
            reduction="mean",
        )

        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + beta * kl_loss

        recon_ppl = torch.exp(recon_loss.detach())

        padding_mask = target != self.net.vocab.pad_idx
        acc = (logits.argmax(-1)[padding_mask] == target[padding_mask]).float().mean().item()

        loss_info = {
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "recon_ppl": recon_ppl,
            "acc": acc,
        }

        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

        for k, v in loss_info.items():
            self.log(f"val/{k}", v, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return {"loss": loss, **loss_info}

    def predict_step(self, batch, batch_idx):
        raise NotImplementedError("Predict step not implemented yet.")

    def configure_optimizers(self):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

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
