from typing import Any, List, Optional, Tuple, Union

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split

from evo.dataset import CollatableDataset


class PLMRDataModule(LightningDataModule):
    """`LightningDataModule` for protein datasets."""

    def __init__(
        self,
        dataset: Optional[CollatableDataset] = None,
        dataset_train: Optional[CollatableDataset] = None,
        dataset_val: Optional[Union[CollatableDataset, List[CollatableDataset]]] = None,
        dataset_test: Optional[Union[CollatableDataset, List[CollatableDataset]]] = None,
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
        self.save_hyperparameters(
            logger=False, ignore=["dataset", "dataset_train", "dataset_val", "dataset_test"]
        )

        assert (
            dataset is not None or dataset_train is not None
        ), "Either 'dataset' or 'dataset_train' must be provided."

        self.dataset: Optional[Dataset] = dataset
        self.data_train: Optional[Dataset] = dataset_train

        # Convert single datasets to lists for uniform handling
        if dataset_val is None:
            self.data_val: List[Dataset] = []
        elif isinstance(dataset_val, list):
            self.data_val: List[Dataset] = dataset_val
        else:
            self.data_val: List[Dataset] = [dataset_val]

        if dataset_test is None:
            self.data_test: List[Dataset] = []
        elif isinstance(dataset_test, list):
            self.data_test: List[Dataset] = dataset_test
        else:
            self.data_test: List[Dataset] = [dataset_test]

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
            # Only perform splitting if dataset_train is not provided
            if not self.data_train:
                train_split, val_split = random_split(
                    dataset=self.dataset,
                    lengths=self.hparams.train_val_split,
                    generator=torch.Generator().manual_seed(self.hparams.generator_seed),
                )
                self.data_train = train_split
                # Insert the split validation set at the beginning
                self.data_val.insert(0, val_split)
        elif stage in ("predict", "test") and not self.data_test:
            self.data_test = [self.dataset]
            if hasattr(self.dataset, "training"):
                self.dataset.training = False

    def _dataloader_template(self, dataset: Dataset[Any], training=True) -> DataLoader[Any]:
        collate_fn = dataset.collater if hasattr(dataset, "collater") else self.collate_fn
        sampler = dataset.sampler() if hasattr(dataset, "sampler") else None
        num_workers = self.hparams.num_workers if training else 0
        pin_memory = self.hparams.pin_memory if training else False
        shuffle = self.hparams.shuffle and training and sampler is None
        return DataLoader(
            dataset=dataset,
            collate_fn=collate_fn,
            batch_size=self.batch_size_per_device,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=shuffle,
            sampler=sampler,
        )

    def train_dataloader(self) -> DataLoader[Any]:
        return self._dataloader_template(self.data_train)

    def val_dataloader(self) -> Union[DataLoader[Any], List[DataLoader[Any]]]:
        if len(self.data_val) == 1:
            return self._dataloader_template(self.data_val[0], training=False)
        return [self._dataloader_template(dataset, training=False) for dataset in self.data_val]

    def test_dataloader(self) -> Union[DataLoader[Any], List[DataLoader[Any]]]:
        if len(self.data_test) == 1:
            return self._dataloader_template(self.data_test[0], training=False)
        return [self._dataloader_template(dataset, training=False) for dataset in self.data_test]


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
