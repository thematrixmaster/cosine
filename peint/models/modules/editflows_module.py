from typing import Tuple, Callable, List
from collections import defaultdict
from functools import partial

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from transformers.optimization import get_polynomial_decay_schedule_with_warmup

from peint.models.frameworks.dfm import *  # noqa
from peint.models.frameworks.editflows import *  # noqa
from peint.models.modules.plmr_module import PLMRLitModule
from peint.models.modules.utils import log_wandb_scatter

from evo.tokenization import Vocab
from evo.phylogeny import VALID_BINS, get_quantile_idx


class EditFlowsModule(PLMRLitModule):
    """Lightning module for training an Edit Flows model from [Havasi et al.](https://arxiv.org/abs/2506.09018)"""

    def __init__(
        self,
        net: nn.Module,
        vocab: Vocab,
        coupling: Coupling,
        kappa_scheduler: KappaScheduler,
        alignment_factory: Callable[..., Alignment],
        sampler_factory: Callable[..., DiscreteSampler],
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

        self.vocab = vocab
        self.pad_token_id = vocab.pad_idx
        self.bos_token_id = vocab.bos_idx
        self.eos_token_id = vocab.eos_idx

        # self.gap_token_id = len(vocab)
        # self.x_vocab_size = len(vocab)
        self.gap_token_id = vocab.tokens_to_idx["-"]
        self.x_vocab_size = len(vocab) - 2
        self.special_token_ids = [
            self.bos_token_id,
            self.pad_token_id,
            self.eos_token_id,
            self.vocab.unk_idx,
        ]
        self.special_aa_token_ids = [vocab.tokens_to_idx[aa] for aa in ["B", "O", "U", "X", "Z"]]

        self.z_vocab_size = self.x_vocab_size + 1  # +1 for gap token

        # Instantiate the partial alignment class
        self.coupling = coupling
        self.kappa_scheduler = kappa_scheduler
        # self.alignment = alignment_factory(vocab=vocab, gap_token_id=self.gap_token_id)
        self.sampler_factory = sampler_factory
        self.avoid_special_tokens_at_sampling = True
        self.avoid_special_amino_acids_at_sampling = True

    def forward(
        self,
        xt: Tensor,
        t: Tensor,
        x_pad_mask: Tensor,
        S=None,
        avoid_special_tokens=False,
        avoid_special_aa_tokens=False,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        out_dict = self.net(xt, t, ~x_pad_mask, S=S)
        if isinstance(out_dict, dict):
            logits: Tensor = out_dict["logits"]
        else:
            logits = out_dict
        assert logits.ndim == 3, "Logits must have shape (batch_size, seq_len, 2 * vocab_size + 3)"
        assert (
            logits.shape[2] == 2 * self.x_vocab_size + 3
        ), "Logits must have shape (batch_size, seq_len, 2 * vocab_size + 3)"
        ut = F.softplus(logits[:, :, :3])
        ins_logits, sub_logits = (
            logits[:, :, 3 : self.x_vocab_size + 3],
            logits[:, :, self.x_vocab_size + 3 :],
        )
        if avoid_special_tokens:
            ins_logits[:, :, self.special_token_ids] = -1e9
            sub_logits[:, :, self.special_token_ids] = -1e9
        if avoid_special_aa_tokens:
            ins_logits[:, :, self.special_aa_token_ids] = -1e9
            sub_logits[:, :, self.special_aa_token_ids] = -1e9
        ins_probs = F.softmax(logits[:, :, 3 : self.x_vocab_size + 3], dim=-1)
        sub_probs = F.softmax(logits[:, :, self.x_vocab_size + 3 :], dim=-1)
        if torch.isnan(ut).any() or torch.isnan(ins_probs).any() or torch.isnan(sub_probs).any():
            raise ValueError("NaN values found in model output logits")
        return ut, ins_probs, sub_probs

    def model_step(self, batch: Tuple[Tensor, ...]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data"""
        xt, t, S, x_pad_mask, z_gap_mask, z_pad_mask, uz_mask, *_ = batch

        # Pass (xt, t) through the net to get rates and probabilities
        ut, ins_probs, sub_probs = self.forward(xt=xt, t=t, x_pad_mask=x_pad_mask, S=S)
        lambda_ins = ut[:, :, 0]
        lambda_sub = ut[:, :, 1]
        lambda_del = ut[:, :, 2]

        # Multiply rates by probs to get per operation rates, then concat
        u_tia_ins = lambda_ins.unsqueeze(-1) * ins_probs
        u_tia_sub = lambda_sub.unsqueeze(-1) * sub_probs
        u_tia_del = lambda_del.unsqueeze(-1)
        ux_cat = torch.cat([u_tia_ins, u_tia_sub, u_tia_del], dim=-1)

        # Convert ux_cat from x space to z space by duplicating logits for gap token indices
        uz_cat = fill_gap_tokens_with_repeats(ux_cat, z_gap_mask, z_pad_mask)

        # Calculate Bregman divergence loss
        u_tot = ut.sum(dim=(1, 2))
        sched_coeff = self.kappa_scheduler.derivative(t) / (1 - self.kappa_scheduler(t))
        log_uz_cat = torch.clamp(uz_cat.log(), min=-20)
        loss = u_tot - (log_uz_cat * uz_mask * sched_coeff.unsqueeze(-1)).sum(dim=(1, 2))
        loss = loss.mean()
        assert not torch.isnan(loss) and not torch.isinf(loss), "Loss is NaN or Inf"

        # Calculate loss information for logging
        loss_info = {
            "t": t.mean().detach().clone(),
            "n_subs": S[:, 0].float().mean().detach().clone(),
            "n_ins": S[:, 1].float().mean().detach().clone(),
            "n_dels": S[:, 2].float().mean().detach().clone(),
            "u_tot": u_tot.mean().detach().clone(),
            "u_ins": lambda_ins.sum(dim=1).mean().detach().clone(),
            "u_sub": lambda_sub.sum(dim=1).mean().detach().clone(),
            "u_del": lambda_del.sum(dim=1).mean().detach().clone(),
            "u_con": (uz_cat * uz_mask).sum(dim=(1, 2)).mean().detach().clone(),
        }
        return loss, loss_info

    def validation_step(self, batch, batch_idx):
        loss_info = super().validation_step(batch, batch_idx)
        if "u_con" in loss_info.keys():
            self.val_ucon(loss_info["u_con"])
        return loss_info

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

    def predict_step(self, batch: Tuple[Tensor, ...], batch_idx) -> List[Tensor]:
        """Sample from the sampler using x0 derived from the batch & coupling"""
        _, _, _, _, _, _, _, x0, x1, S, jump_schedule = batch
        S = S.to(self.device)
        sampler = self.make_sampler()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return sampler(x0.to(self.device), S=S, jump_schedule=jump_schedule)

    def make_sampler(self, **kwargs) -> DiscreteSampler:
        """Create a sampler for the Edit Flows model"""
        self.net.eval()
        _max_seq_len = getattr(self.net, "max_seq_len", 512)
        max_seq_len = kwargs.pop("max_seq_len", _max_seq_len)
        forward_fn = lambda *a, **kw: self.forward(
            *a,
            **kw,
            avoid_special_tokens=self.avoid_special_tokens_at_sampling,
            avoid_special_aa_tokens=self.avoid_special_amino_acids_at_sampling,
        )
        return self.sampler_factory(
            forward_fn=forward_fn,
            pad_token_id=self.pad_token_id,
            scheduler=self.kappa_scheduler,
            max_seq_len=max_seq_len,
            **kwargs,
        )

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
