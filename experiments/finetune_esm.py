import datetime
import os
from argparse import ArgumentParser

import esm
import lightning as pl
import torch
from evo.dataset import (
    EncodedFastaDataset,
    MaskedTokenWrapperDataset,
    RandomCropDataset,
)
from evo.tokenization import Vocab
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DDPStrategy
from peft import LoraConfig, TaskType, get_peft_model
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import MaskedLMOutput

from peint.data.datamodule import PLMRDataModule
from peint.models.nets.esm2 import ESM2Flash

torch.set_float32_matmul_precision("medium")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["WANDB_MODE"] = "disabled"


def get_dataloaders(args, vocab: Vocab):
    dataset = EncodedFastaDataset(
        data_file=args.fasta_path,
        cache_indices=True,
        vocab=vocab,
    )
    dataset = RandomCropDataset(dataset, max_seqlen=1024)
    dataset = MaskedTokenWrapperDataset(
        dataset, mask_prob=0.15, random_token_prob=0.1, leave_unmasked_prob=0.1
    )
    datamodule = PLMRDataModule(
        dataset=dataset,
        batch_size=args.batch_size,
        generator_seed=args.seed,
        train_val_split=(0.995, 0.005),
        num_workers=args.num_workers if hasattr(args, "num_workers") else 0,
        shuffle=True,
    )
    datamodule.setup()
    return datamodule.train_dataloader(), datamodule.val_dataloader()


class ESM2LightningModule(pl.LightningModule):
    def __init__(self, args, vocab: Vocab):
        super().__init__()
        self.save_hyperparameters(args)
        self.args = args

        self.vocab = vocab

        _esm_model, _ = esm.pretrained.esm2_t30_150M_UR50D()
        esm_model = ESM2Flash(
            num_layers=_esm_model.num_layers,
            embed_dim=_esm_model.embed_dim,
            attention_heads=_esm_model.attention_heads,
            alphabet="ESM-1b",
            token_dropout=_esm_model.token_dropout,
        )
        esm_model.load_state_dict(_esm_model.state_dict(), strict=False)
        del _esm_model
        self.model = esm_model

        if args.use_lora:
            lora_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=["query", "key", "value", "dense"],
            )
            self.model = get_peft_model(self.model, lora_config)
            print("LoRA enabled - printing trainable parameters:")
            self.model.print_trainable_parameters()
        else:
            print("Regular fine-tuning enabled - all parameters will be trained")
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"Total parameters: {total_params:,}")
            print(f"Trainable parameters: {trainable_params:,}")
            print(f"Trainable%: {100 * trainable_params / total_params:.2f}%")

    def forward(self, input_ids, attention_mask=None, labels=None):
        logits = self.model(x=input_ids)["logits"]
        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=self.vocab.pad_idx)
            labels = labels.to(logits.device)
            masked_lm_loss = loss_fct(logits.view(-1, len(self.vocab)), labels.view(-1))
        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=logits,
        )

    def training_step(self, batch, batch_idx):
        [input_ids, labels] = batch
        attention_mask = input_ids != self.vocab.pad_idx

        outputs = self(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        [input_ids, labels] = batch
        attention_mask = input_ids != self.vocab.pad_idx

        outputs = self(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        perplexity = torch.exp(loss)

        predictions = outputs.logits.argmax(dim=-1)
        mask_positions = labels != self.vocab.pad_idx
        if mask_positions.sum() > 0:
            accuracy = (predictions[mask_positions] == labels[mask_positions]).float().mean()
        else:
            accuracy = torch.tensor(0.0)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_perplexity", perplexity, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_accuracy", accuracy, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.epochs)
        return [optimizer], [scheduler]


def main(args):
    # Create run directory with timestamp
    run_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(args.log_dir, "runs", run_timestamp)
    os.makedirs(run_dir, exist_ok=True)

    # Create checkpoint directory within run dir
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Load esm2 vocab
    vocab = Vocab.from_esm_alphabet(esm.Alphabet.from_architecture("ESM-1b"))

    # Prepare dataloaders
    train_loader, val_loader = get_dataloaders(args, vocab=vocab)

    # Initialize Lightning module
    lightning_module = ESM2LightningModule(args, vocab=vocab)

    # Setup loggers
    loggers = []

    # Initialize WandbLogger. This should be done for all processes.
    # Lightning takes care of syncing logs and only saving checkpoints on rank 0.
    if (
        args.gpus > 0
    ):  # Only create wandb logger if using GPUs, which are common for distributed training
        wandb_logger = WandbLogger(
            project=args.wandb_project,
            name=args.run_name,
            save_dir=run_dir,
            log_model=False,
            offline=False,
            prefix="",
            entity="thematrixmaster",
            group="",
            tags=["finetune", "esm2"],
            job_type="",
        )
        loggers.append(wandb_logger)

    # csv_logger = CSVLogger(save_dir=run_dir, name="csv")
    # loggers.append(csv_logger)

    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="esm2-{epoch:02d}-{val_loss:.2f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    early_stopping_callback = EarlyStopping(monitor="val_loss", patience=5, mode="min")

    # EMA callback can be added if desired
    # ema_callback = EMA(0.9999)

    # Setup distributed strategy for multi-GPU training
    # PyTorch Lightning automatically handles the best strategy
    # The 'auto' strategy is often sufficient. If using DDP, find_unused_parameters is often a bad idea
    # unless you explicitly need it, as it can be slow.
    strategy = "auto"
    if args.gpus > 1:
        strategy = DDPStrategy(find_unused_parameters=True)

    # Initialize PyTorch Lightning trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if torch.cuda.is_available() and args.gpus > 0 else "cpu",
        devices=args.gpus if torch.cuda.is_available() and args.gpus > 0 else "auto",
        strategy=strategy,
        logger=loggers,
        callbacks=[checkpoint_callback, early_stopping_callback],
        gradient_clip_val=args.gradient_clip_val,
        accumulate_grad_batches=args.accumulate_grad_batches,
        precision=args.precision,
        val_check_interval=args.val_check_interval,
    )

    # Start training
    trainer.fit(lightning_module, train_loader, val_loader, ckpt_path=args.checkpoint_path)


if __name__ == "__main__":
    parser = ArgumentParser()

    # Data arguments
    parser.add_argument(
        "--fasta_path",
        type=str,
        default="/scratch/users/stephen.lu/projects/protevo/data/oas/antiref/antiref90.fasta",
        help="Path to the input FASTA file.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for data loading.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )

    # LoRA arguments
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="Enable LoRA fine-tuning (default: False, regular fine-tuning).",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=16,
        help="LoRA rank parameter.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LoRA alpha parameter.",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.1,
        help="LoRA dropout rate.",
    )

    # Training arguments
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate for training.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay for optimizer.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="Number of GPUs to use for training (supports multi-GPU with DDP).",
    )
    parser.add_argument(
        "--gradient_clip_val",
        type=float,
        default=1.0,
        help="Gradient clipping value.",
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="Number of batches to accumulate gradients over.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="bf16-mixed",
        help="Training precision",
    )
    parser.add_argument(
        "--val_check_interval",
        type=float,
        default=1.0,
        help="How often to check validation loss within an epoch.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of worker threads for data loading.",
    )

    # Logging and checkpointing arguments
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="esm2-finetuning",
        help="WandB project name.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="WandB run name. If None, will be auto-generated.",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="/accounts/projects/yss/stephen.lu/protevo/plmr/logs/esm_finetune",
        help="Base directory for logs and checkpoints.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to a checkpoint to resume training from.",
    )

    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)

    if args.run_name is None:
        lora_suffix = "lora" if args.use_lora else "full"
        args.run_name = f"esm2-{lora_suffix}-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"

    assert os.path.exists(args.fasta_path), f"FASTA file not found: {args.fasta_path}"

    main(args)
