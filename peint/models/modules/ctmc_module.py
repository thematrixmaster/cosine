from collections import defaultdict
from functools import partial
from typing import List

import lightning as pl
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from lightning.pytorch.loggers import WandbLogger
from torch import Tensor
from transformers.optimization import get_polynomial_decay_schedule_with_warmup

from evo.phylogeny import VALID_BINS, get_quantile_idx
from peint.models.modules.plmr_module import PLMRLitModule
from peint.models.nets.ctmc import NeuralCTMC


def log_wandb_scatter(
    trainer: pl.Trainer,
    arr_per_bin: np.ndarray,
    name: str = "time_bin_likelihood",
    xlabel: str = "Time",
    ylabel: str = "Mean per-site Likelihood",
    title: str = "Loglikelihood per time bin",
    transform_fn: callable = lambda v: np.exp(np.mean(v)),
):
    arrmeans = np.zeros(len(VALID_BINS))
    for k, v in arr_per_bin.items():
        arrmeans[k] = transform_fn(v)

    nonzero = np.nonzero(arrmeans)[0]
    xs = VALID_BINS[nonzero]
    ys = arrmeans[nonzero]

    data = [[x, y] for x, y in zip(xs, ys)]
    table = wandb.Table(data=data, columns=[xlabel, ylabel])

    # important for multi-gpu runs, doesn't seem to affect single gpu runs
    if trainer.global_rank == 0 and isinstance(trainer.logger, WandbLogger):
        trainer.logger.experiment.log(
            {
                name: wandb.plot.scatter(
                    table,
                    xlabel,
                    ylabel,
                    title=title,
                )
            }
        )


class CTMCModule(PLMRLitModule):
    """Unified Lightning module for training CTMC model"""

    def __init__(
        self,
        net: NeuralCTMC,
        weight_decay: float = 0.01,
        optimizer: torch.optim.Optimizer = partial(torch.optim.AdamW, lr=1e-4),
        scheduler: torch.optim.lr_scheduler = partial(
            get_polynomial_decay_schedule_with_warmup,
            num_warmup_steps=10000,
            num_training_steps=100000,
            power=2.0,
        ),
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

    def forward(self, x, t, x_sizes, Q=None, *args, **kwargs):
        """Obtain per-site rate matrices and obtain log likelihoods for rows in x"""
        if Q is None:
            Q = self.net(x, x_sizes, *args, **kwargs)  # (B, L, V, V)
        t = t.unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1, 1)
        P = torch.matrix_exp(Q * t)  # (B, L, V, V)
        # we want to index into P[b, l, x[b,l], :] to get a tensor of shape (B, L, V)
        batch_indices = torch.arange(x.size(0), device=x.device).unsqueeze(-1)  # (B, 1)
        seq_indices = torch.arange(x.size(1), device=x.device).unsqueeze(0)  # (1, L)
        P_selected = P[batch_indices, seq_indices, x]  # (B, L, V)
        log_probs = torch.log(P_selected + 1e-8)  # (B, L, V)
        return log_probs

    def model_step(self, batch):
        x, y, t, x_sizes = batch

        # Use unified network interface with size-based masks
        log_probs: Tensor = self.forward(x, t, x_sizes=x_sizes)  # (B, L, V)

        # compute cross entropy loss
        nll = self.net.criterion(log_probs.transpose(-1, -2), y)  # (B, L)

        # compute perplexity
        ppl = torch.exp(nll.detach())

        return nll, dict(nll=nll, ppl=ppl)

    def validation_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        """Simplified validation step using unified network interface"""
        x, y, t, x_sizes = batch

        with torch.no_grad():
            Q: Tensor = self.net(x, x_sizes=x_sizes)  # (B, L, V, V)
            log_probs: Tensor = self.forward(x, t, x_sizes=x_sizes, Q=Q)  # (B, L, V)

        # Keep unreduced to get per-site time likelihood (B, L)
        nll = F.cross_entropy(
            log_probs.transpose(-1, -2),
            y,
            ignore_index=self.net.vocab.pad_idx,
            reduction="none",
        ).detach()

        # compute perplexity
        padding_mask = y != self.net.vocab.pad_idx
        mean_nll = nll[padding_mask].mean()
        ppl = torch.exp(mean_nll)

        # get log-likelihoods per time bin
        tbins = t.expand_as(y)[padding_mask]
        ll_flat = -1 * nll[padding_mask]
        ll_per_bin = {b.item(): ll_flat[tbins == b].cpu().numpy() for b in t}

        # compute transition entropy per time bin
        entropy = -(log_probs * log_probs.exp()).sum(dim=-1)  # (B, L)
        entropy_flat = entropy[padding_mask]

        # find min and max off-diagonal entries in Q per time bin
        V = len(self.net.vocab)
        off_diag_mask = ~torch.eye(V, dtype=bool, device=Q.device)  # (V, V)
        min_q_flat = Q[:, :, off_diag_mask].min(dim=-1).values[padding_mask]
        max_q_flat = Q[:, :, off_diag_mask].max(dim=-1).values[padding_mask]

        # add to per-bin stats
        for b in t:
            k = get_quantile_idx(VALID_BINS, b.item())
            self.min_q_per_bin[k].extend(min_q_flat[tbins == b].cpu().numpy())
            self.max_q_per_bin[k].extend(max_q_flat[tbins == b].cpu().numpy())
            self.h_per_bin[k].extend(entropy_flat[tbins == b].cpu().numpy())

        loss = mean_nll
        acc = (log_probs.argmax(-1)[padding_mask] == y[padding_mask]).float().mean().item()

        loss_info = {
            "nll_loss": mean_nll,
            "ppl": ppl,
            "acc": acc,
        }

        self.val_loss(mean_nll)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

        for k, v in loss_info.items():
            self.log(f"val/{k}", v, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return {"loss": loss, **loss_info}, ll_per_bin

    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()
        self.h_per_bin = defaultdict(list)
        self.min_q_per_bin = defaultdict(list)
        self.max_q_per_bin = defaultdict(list)

    def on_validation_epoch_end(self, *args, **kwargs) -> None:
        super().on_validation_epoch_end(*args, **kwargs)
        log_wandb_scatter(
            self.trainer,
            self.h_per_bin,
            name="transition-ppl",
            xlabel="Time",
            ylabel="Mean Transition Perplexity",
            title="Transition Perplexity per time bin",
            transform_fn=lambda v: np.exp(np.mean(v)),
        )
        log_wandb_scatter(
            self.trainer,
            self.min_q_per_bin,
            name="min-rate",
            xlabel="Time",
            ylabel="Min Off-diagonal Rate",
            title="Min Off-diagonal Rate per time bin",
            transform_fn=lambda v: np.mean(v),
        )
        log_wandb_scatter(
            self.trainer,
            self.max_q_per_bin,
            name="max-rate",
            xlabel="Time",
            ylabel="Max Off-diagonal Rate",
            title="Max Off-diagonal Rate per time bin",
            transform_fn=lambda v: np.mean(v),
        )

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
