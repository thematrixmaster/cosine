from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

import torch
from lightning import LightningModule
from torch import Tensor
from torchmetrics import MeanMetric, MinMetric


class PLMRLitModule(LightningModule, ABC):
    """Lightning module base class for training a PLMR model"""

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        ignore: List[str] = [],
        *args,
        **kwargs,
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=ignore)

        self.net = net
        # can_pickle_net(net) # ensure that the net module is pickleable

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation loss
        self.val_loss_best = MinMetric()

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.net(x, **kwargs)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins"""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_loss_best.reset()

    @abstractmethod
    def model_step(self, batch: Dict[str, Tensor]) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Returns loss and loss info dictionary for a given batch"""
        raise NotImplementedError()

    @abstractmethod
    def predict_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tensor:
        """Make predictions on a batch of data from the dataset."""
        raise NotImplementedError()

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tensor:
        """Perform a single training step on a batch of data from the training set"""
        loss, loss_info = self.model_step(batch)

        # update and log training loss
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)

        # log additional loss info
        for k, v in loss_info.items():
            self.log(f"train/{k}", v, on_step=True, on_epoch=False, prog_bar=True, sync_dist=False)

        return loss

    def on_train_epoch_end(self) -> None:
        """Lightning hook that is called when a training epoch ends"""

    def validation_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Dict[str, Tensor]:
        """Perform a single validation step on a batch of data from the validation set"""
        loss, loss_info = self.model_step(batch)

        # update and log validation loss
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

        # log additional loss info
        for k, v in loss_info.items():
            self.log(f"val/{k}", v, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return {"loss": loss, **loss_info}

    def on_validation_epoch_end(self) -> None:
        """Lightning hook that is called when a validation epoch ends"""
        loss = self.val_loss.compute()
        self.val_loss_best(loss)
        self.log("val/loss_best", self.val_loss_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Dict[str, Tensor]:
        """Perform a single test step on a batch of data from the test set."""
        loss, loss_info = self.model_step(batch)

        # update and log test loss
        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)

        # log additional loss info
        for k, v in loss_info.items():
            self.log(f"test/{k}", v, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return {"loss": loss, **loss_info}

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends"""

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "train/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
