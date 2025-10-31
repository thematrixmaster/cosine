from collections import defaultdict
from functools import partial
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from transformers.optimization import get_polynomial_decay_schedule_with_warmup

from evo.phylogeny import VALID_BINS, get_quantile_idx
from peint.models.modules.plmr_module import PLMRLitModule
from peint.models.modules.utils import log_wandb_scatter
from peint.models.nets.lept import LEPT


class LEPTModule(PLMRLitModule):
    net: LEPT

    def __init__(
        self,
        net: LEPT,
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

        loss_dict = self.net.loss(x=x, y=y, x_sizes=x_sizes, y_sizes=y_sizes, t=t, beta=beta)

        loss = loss_dict["loss"]

        recon_loss = loss_dict["recon_loss"]
        recon_ppl = torch.exp(recon_loss.detach())

        trans_loss = loss_dict["trans_dec_loss"]
        trans_ppl = torch.exp(trans_loss.detach())

        loss_dict.pop("loss", None)
        metrics = {
            **loss_dict,
            "recon_ppl": recon_ppl,
            "trans_ppl": trans_ppl,
            "beta": beta,
        }
        return loss, metrics

    def validation_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        x, y, t, x_sizes, y_sizes = batch
        beta = self.beta_max

        # reconstruction objective
        with torch.no_grad():
            recon_loss_dict = self.net.recon_loss(x, y, x_sizes, y_sizes, calc_acc=True, beta=beta)

        recon_loss = recon_loss_dict["recon_loss"]
        recon_ppl = torch.exp(recon_loss.detach())

        # transition objective
        with torch.no_grad():
            Z_x = self.net.encode(x, x_sizes)
            Z_y = self.net.encode(y, y_sizes)
            Z_y_hat = self.net.exp(Z_x, t, num_steps=self.net.num_steps)
            y_attn_mask = (y != self.net.vocab.pad_idx).long()
            logits = self.net.decoder(y[:, :-1], Z_y_hat, y_attn_mask[:, :-1])

        trans_l2_dist = torch.norm(Z_y_hat - Z_y, dim=-1, p=2).unsqueeze(1)  # (bs,1)
        trans_lat_loss = F.mse_loss(input=Z_y_hat, target=Z_y, reduction="none").mean(
            -1, keepdim=True
        )  # (bs,1)
        trans_dec_loss = F.cross_entropy(
            logits.transpose(1, 2),
            y[:, 1:],
            ignore_index=self.net.vocab.pad_idx,
            reduction="none",
        )  # (bs, seq_len-1)

        # stratify loss by time t into bins
        y_tgt_attn_mask = y_attn_mask[:, :-1].bool()
        tbins = t.expand_as(y[:, :-1])[y_tgt_attn_mask]
        trans_ll = -trans_dec_loss[y_tgt_attn_mask]

        for idx, b in enumerate(t):
            k = get_quantile_idx(VALID_BINS, b.item())
            self.trans_ll_per_bin[k].extend(trans_ll[tbins == b].cpu().numpy())
            self.trans_mse_per_bin[k].extend(trans_lat_loss[idx].cpu().numpy())
            self.trans_l2_per_bin[k].extend(trans_l2_dist[idx].cpu().numpy())

        # aggregate losses and metrics
        trans_lat_loss = trans_lat_loss.mean()
        trans_dec_loss = trans_dec_loss[y_tgt_attn_mask].mean()
        loss = recon_loss + trans_lat_loss + trans_dec_loss
        trans_ppl = torch.exp(trans_dec_loss.detach())
        trans_acc = (
            (logits.argmax(-1)[y_tgt_attn_mask] == y[:, 1:][y_tgt_attn_mask]).float().mean().item()
        )

        recon_loss_dict.pop("loss", None)
        loss_info = {
            **recon_loss_dict,
            "trans_lat_loss": trans_lat_loss.detach().item(),
            "trans_dec_loss": trans_dec_loss.detach().item(),
            "recon_ppl": recon_ppl,
            "trans_ppl": trans_ppl,
            "trans_acc": trans_acc,
            "beta": beta,
        }

        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

        for k, v in loss_info.items():
            self.log(f"val/{k}", v, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return {"loss": loss, **loss_info}

    def on_validation_epoch_start(self, *args, **kwargs):
        super().on_validation_epoch_start(*args, **kwargs)
        self.trans_ll_per_bin = defaultdict(list)
        self.trans_mse_per_bin = defaultdict(list)
        self.trans_l2_per_bin = defaultdict(list)

    def on_validation_epoch_end(self, *args, **kwargs):
        super().on_validation_epoch_end(*args, **kwargs)
        log_wandb_scatter(
            self.trainer,
            self.trans_ll_per_bin,
            name="transition-ll",
            xlabel="Time",
            ylabel="Mean Transition Log-Likelihood",
            title="Transition ll (in data space) per time bin",
            transform_fn=lambda v: np.mean(v),
        )
        log_wandb_scatter(
            self.trainer,
            self.trans_mse_per_bin,
            name="transition-mse",
            xlabel="Time",
            ylabel="Mean Transition MSE",
            title="Transition MSE (in latent space) per time bin",
            transform_fn=lambda v: np.mean(v),
        )
        log_wandb_scatter(
            self.trainer,
            self.trans_l2_per_bin,
            name="transition-l2",
            xlabel="Time",
            ylabel="Mean Transition L2 Distance",
            title="Transition L2 Distance (in latent space) per time bin",
            transform_fn=lambda v: np.mean(v),
        )

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
