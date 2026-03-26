import os
import sys
from collections import defaultdict
from functools import partial
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.optimization import get_polynomial_decay_schedule_with_warmup

from evo.phylogeny import VALID_BINS, get_quantile_idx
from peint.models.modules.plmr_module import PLMRLitModule
from peint.models.modules.utils import log_wandb_scatter
from peint.models.nets.peint import PEINT, ESMEncoder


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

    def forward(self, *args, **kwargs):
        """Forward method delegates to unified network interface"""
        return self.net(*args, **kwargs)

    def model_step(self, batch):
        # New unified format: [x_src, x_tgt, y_src, y_tgt, ts, chain_ids, x_sizes, y_sizes]
        x_src, x_tgt, y_src, y_tgt, ts, chain_ids, x_sizes, y_sizes = batch

        # Use unified network interface with size-based masks
        outputs = self.net(x_src, y_src, ts, x_sizes=x_sizes, y_sizes=y_sizes, chain_ids=chain_ids)

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

    def validation_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        """Simplified validation step using unified network interface"""
        # Handle different data formats
        if dataloader_idx == 0:
            # Standard training format has x_tgt
            x_src, x_tgt, y_src, y_tgt, ts, chain_ids, x_sizes, y_sizes = batch
        else:
            # DMS format only cares about decoder prediction
            x_src, y_src, y_tgt, muts, ts, chain_ids, x_sizes, y_sizes = batch
            x_tgt = None

        with torch.no_grad():
            outputs = self.net(
                x_src, y_src, ts, x_sizes=x_sizes, y_sizes=y_sizes, chain_ids=chain_ids
            )

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

        # get lls per time bin
        tbins = ts[:, :1].expand_as(y_tgt)[padding_mask]
        ll_per_site = -1 * clm_loss[padding_mask]
        for b in ts[:, 0]:
            k = get_quantile_idx(VALID_BINS, b.item())
            self.ll_per_bin[k].extend(ll_per_site[tbins == b].cpu().numpy())

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

    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()
        self.ll_per_bin = defaultdict(list)

    def on_validation_epoch_end(self, *args, **kwargs) -> None:
        super().on_validation_epoch_end(*args, **kwargs)
        log_wandb_scatter(
            self.trainer,
            self.ll_per_bin,
            name="ll-per-bin",
            xlabel="Time",
            ylabel="Mean Log-Likelihood",
            title="LL per time bin",
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


def load_from_old_checkpoint(checkpoint_path: str, device: str = "cpu") -> PEINTModule:
    """
    Load a PEINTModule from an old checkpoint format.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file {checkpoint_path} does not exist.")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    hyperparams = checkpoint.get("hyper_parameters", {})

    enc_model = ESMEncoder.from_pretrained(
        dropout_p=hyperparams.get("dropout_p", 0.0),
        finetune=False,
        embed_x_per_chain=False,
    )
    peint = PEINT(
        enc_model=enc_model,
        evo_vocab=enc_model.vocab,
        num_chains=1,
        causal_decoder=True,
        use_chain_embedding=False,
        num_heads=hyperparams["num_heads"],
        embed_dim=hyperparams["embed_dim"],
        max_len=hyperparams["max_seq_len"],
        num_encoder_layers=hyperparams["num_encoder_layers"],
        num_decoder_layers=hyperparams["num_decoder_layers"],
        dropout_p=hyperparams.get("dropout_p", 0.0),
        use_attention_bias=hyperparams.get("use_attention_bias", True),
    )

    module = PEINTModule(
        net=peint,
        weight_decay=hyperparams.get("weight_decay", 0.0),
        compile=hyperparams.get("compile", False),
    )

    state_dict = {}
    for k, v in checkpoint["state_dict"].items():
        if k.startswith("model.esm"):
            continue
        assert k.startswith("model.")
        state_dict[k.replace("model.", "net.")] = v

    missing_keys, unexpected_keys = module.load_state_dict(state_dict=state_dict)
    assert all(k.startswith("net.esm") for k in missing_keys)
    assert len(unexpected_keys) == 0

    return module.to(device).eval()


def load_from_og_peint_checkpoint(checkpoint_path: str, device: str = "cpu") -> PEINTModule:
    """Load checkpoint from original protein-evolution codebase (ProtEvoPretrainedTransformerModule).

    This loader handles checkpoints trained with the old protein-evolution codebase, which used
    a different model structure and state dict format. Key adaptations:
    - Maps old state dict keys (model.* → net.*)
    - Configures for single-sequence mode (num_chains=1)
    - Disables multi-sequence features (chain embeddings)

    Args:
        checkpoint_path: Path to .ckpt file from protein-evolution training
        device: Device to load model onto ('cpu', 'cuda', etc.)

    Returns:
        PEINTModule ready for inference in eval mode

    Example:
        >>> module = load_from_og_peint_checkpoint("checkpoint.ckpt", device="cuda")
        >>> vocab = module.net.vocab
        >>>
        >>> # Encode sequence (vocab.encode adds BOS/EOS automatically)
        >>> x = torch.from_numpy(vocab.encode("MKTAY")).unsqueeze(0).cuda()
        >>> seq_len = x.shape[1]
        >>>
        >>> # Create input tensors
        >>> x_sizes = torch.zeros((1, seq_len), dtype=torch.long, device="cuda")
        >>> x_sizes[0, 0] = seq_len
        >>> t = torch.tensor([[0.1]], dtype=torch.float32, device="cuda")
        >>>
        >>> # Run inference (use bfloat16 for FlashAttention)
        >>> with torch.no_grad():
        >>>     with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        >>>         outputs = module.net(x, x.clone(), t, x_sizes, x_sizes)
    """
    import esm
    from peint.models.nets.esm2 import ESM2Flash

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    hparams = ckpt["hyper_parameters"]

    # Create ESM encoder matching checkpoint architecture
    esm_vocab_obj = esm.data.Alphabet.from_architecture("ESM-1b")
    from evo.tokenization import Vocab

    vocab = Vocab.from_esm_alphabet(esm_vocab_obj)

    flash_esm = ESM2Flash(
        num_layers=30,
        embed_dim=640,
        attention_heads=20,
        alphabet="ESM-1b",
        token_dropout=True,
        dropout_p=hparams.get("dropout_p", 0.0),
    )

    enc_model = ESMEncoder(vocab=vocab, esm_model=flash_esm, finetune=False, embed_x_per_chain=False)

    # Create PEINT model (single-sequence, no chain embeddings)
    net = PEINT(
        enc_model=enc_model,
        evo_vocab=vocab,
        embed_dim=hparams["embed_dim"],
        num_heads=hparams["num_heads"],
        num_chains=1,
        num_encoder_layers=hparams["num_encoder_layers"],
        num_decoder_layers=hparams["num_decoder_layers"],
        max_len=hparams.get("max_seq_len", 1022),
        dropout_p=hparams.get("dropout_p", 0.0),
        use_chain_embedding=False,
        use_attention_bias=hparams.get("use_attention_bias", True),
        causal_decoder=True,
    )

    # Map old state dict keys to new format
    new_state_dict = {}
    for key, value in ckpt["state_dict"].items():
        if not key.startswith("model."):
            continue
        new_key = key[6:]  # Remove 'model.' prefix
        if new_key.startswith("esm."):
            new_key = "enc_model." + new_key
        elif new_key.startswith("embedding."):
            new_key = new_key.replace("embedding.", "in_embedding.")
        elif new_key.startswith("lm_head."):
            new_key = new_key.replace("lm_head.", "out_lm_head.")
        new_state_dict[new_key] = value

    net.load_state_dict(new_state_dict, strict=False)

    # Wrap in Lightning module
    module = PEINTModule(net=net, weight_decay=hparams.get("weight_decay", 0.01))

    return module.to(device).eval()


def load_from_new_checkpoint(checkpoint_path: str, device: str = "cpu") -> PEINTModule:
    """
    Load a PEINTModule from a checkpoint that contains the PEINT model.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file {checkpoint_path} does not exist.")

    import peint

    sys.modules["plmr"] = peint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    hyperparams = checkpoint.get("hyper_parameters", {})
    net = hyperparams.get("net")

    _net_type = type(net).__name__
    if _net_type != "PEINT":
        raise ValueError("Checkpoint does not contain a valid PEINT model.")

    if net.finetune_esm:
        module = PEINTModule.load_from_checkpoint(checkpoint_path=checkpoint_path, strict=True)
    else:
        peint = PEINT.from_pretrained_esm2(
            embed_dim=net.embed_dim,
            num_heads=net.num_heads,
            num_encoder_layers=net.num_encoder_layers,
            num_decoder_layers=net.num_decoder_layers,
            max_len=net.max_len,
            dropout_p=net.dropout_p,
            use_attention_bias=net.use_bias,
            finetune_esm=net.finetune_esm,
        )
        module = PEINTModule.load_from_checkpoint(checkpoint_path, net=peint, strict=False)

    del sys.modules["plmr"]
    return module.to(device).eval()
