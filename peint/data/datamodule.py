from typing import Any, Optional, Tuple

import torch
from evo.dataset import CollatableDataset
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split


class PLMRDataModule(LightningDataModule):
    """`LightningDataModule` for protein datasets."""

    def __init__(
        self,
        dataset: Optional[CollatableDataset] = None,
        dataset_train: Optional[CollatableDataset] = None,
        dataset_val: Optional[CollatableDataset] = None,
        dataset_test: Optional[CollatableDataset] = None,
        batch_size: int = 64,
        generator_seed: int = 42,
        train_val_split: Tuple[float, float] = (0.95, 0.05),
        num_workers: int = 0,
        pin_memory: bool = False,
        shuffle: bool = False,
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        assert (
            dataset is not None or dataset_train is not None
        ), "Either 'dataset' or 'dataset_train' must be provided."

        self.dataset: Optional[Dataset] = dataset
        self.data_train: Optional[Dataset] = dataset_train
        self.data_val: Optional[Dataset] = dataset_val
        self.data_test: Optional[Dataset] = dataset_test

        self.collate_fn = dataset.collater if hasattr(dataset, "collater") else None
        self.batch_size_per_device = batch_size

    def setup(self, stage: Optional[str] = None) -> None:
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size
        if stage is None or stage == "fit":
            if not self.data_train and not self.data_val:
                self.data_train, self.data_val = random_split(
                    dataset=self.dataset,
                    lengths=self.hparams.train_val_split,
                    generator=torch.Generator().manual_seed(self.hparams.generator_seed),
                )
        elif stage in ("predict", "test") and not self.data_test:
            self.data_test = self.dataset
            self.data_test.training = False

    def _dataloader_template(self, dataset: Dataset[Any], training=True) -> DataLoader[Any]:
        collate_fn = dataset.collater if hasattr(dataset, "collater") else self.collate_fn
        sampler = dataset.sampler() if hasattr(dataset, "sampler") else None
        return DataLoader(
            dataset=dataset,
            collate_fn=collate_fn,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=self.hparams.shuffle and training,
            sampler=sampler,
        )

    def train_dataloader(self) -> DataLoader[Any]:
        return self._dataloader_template(self.data_train)

    def val_dataloader(self) -> DataLoader[Any]:
        return self._dataloader_template(self.data_val, training=False)

    def test_dataloader(self) -> DataLoader[Any]:
        return self._dataloader_template(self.data_test, training=False)


# Usage
if __name__ == "__main__":
    from esm.data import Alphabet
    from evo.dataset import (
        EncodedMSADataset,
        EncodedParquetDataset,
        RandomCropDataset,
        SubsampleMSADataset,
    )
    from evo.tokenization import Vocab

    vocab = Vocab.from_esm_alphabet(Alphabet.from_architecture("msa_transformer"))

    if 1:
        dataset = EncodedMSADataset(
            data_file="data/proteingym/DMS_msa_files",
            file_ext="a2m",
            vocab=vocab,
        )
        dataset = RandomCropDataset(dataset, max_seqlen=1024)
        dataset = SubsampleMSADataset(dataset, max_tokens=16384, max_seqs=1024)
    if 0:
        dataset = EncodedParquetDataset(
            data_file="data/sets/b1lpa6_ecosm_russ_2020.parquet",
            sequence_col="sequence",
            prop_cols=["fitness"],
            vocab=vocab,
        )

    datamodule = PLMRDataModule(
        dataset=dataset,
        batch_size=8,
        num_workers=0,
        pin_memory=False,
    )

    # Setup the datamodule
    datamodule.setup()

    # Print dataset sizes
    print(f"Train dataset size: {len(datamodule.data_train)}")
    print(f"Validation dataset size: {len(datamodule.data_val)}")

    # Print first sample from each dataset
    print("First train sample:", datamodule.data_train[0])
    print("First validation sample:", datamodule.data_val[0])

    # Get a batch from each dataloader and print it
    train_loader = datamodule.train_dataloader()
    for batch in train_loader:
        print("Train batch:", batch)
        break

    val_loader = datamodule.val_dataloader()
    for batch in val_loader:
        print("Validation batch:", batch)
        break
