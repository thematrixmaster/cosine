import numpy as np
import torch

from evo.dataset import ComplexCherriesDataset, TorchWrapperDataset
from evo.tensor import collate_tensors
from evo.tokenization import Vocab


class EncodedLEPTDataset(TorchWrapperDataset):
    def __init__(
        self,
        dataset: ComplexCherriesDataset,
        vocab: Vocab,
        swap_prob: float = 0.5,
        sep_token: str = ".",
        *args,
        **kwargs,
    ):
        super().__init__(dataset=dataset, vocab=vocab, *args, **kwargs)
        self.swap_prob = swap_prob
        self.sep_token = sep_token

    def __getitem__(self, index: int):
        xs, ys, ts, _ = super().__getitem__(index)
        ts = ts[0]  # Only use single time value

        # Random swap between parent (xs) and child (ys)
        if np.random.rand() < self.swap_prob:
            xs, ys = ys, xs

        # Encode x with separator tokens between chains
        x_sizes = []
        for i, chain in enumerate(xs):
            size = len(chain)
            if i == 0:
                size += self.vocab.prepend_bos
            if i < len(xs) - 1:
                size += 1  # Add separator token length
            if i == len(xs) - 1:
                size += self.vocab.append_eos
            x_sizes.append(size)

        x_sizes = torch.tensor(x_sizes, dtype=torch.long)
        x_joined = self.sep_token.join(xs)
        x = torch.from_numpy(self.vocab.encode_single_sequence(x_joined))

        # Encode y with separator tokens between chains
        y_sizes = []
        for i, chain in enumerate(ys):
            size = len(chain)
            if i == 0:
                size += self.vocab.prepend_bos
            if i < len(ys) - 1:
                size += 1  # Add separator token length
            if i == len(ys) - 1:
                size += self.vocab.append_eos
            y_sizes.append(size)

        y_sizes = torch.tensor(y_sizes, dtype=torch.long)
        y_joined = self.sep_token.join(ys)
        y = torch.from_numpy(self.vocab.encode_single_sequence(y_joined))

        # Handle time values
        if isinstance(ts, (int, float)):
            ts = torch.tensor([ts], dtype=torch.float32)
        elif all(isinstance(t, (int, float)) for t in ts):
            ts = torch.tensor(ts, dtype=torch.float32)
        else:
            ts = np.array(ts, dtype=object)

        # Pad sizes to match sequence length
        x_sizes = torch.nn.functional.pad(x_sizes, (0, len(x) - len(x_sizes)), value=0)
        y_sizes = torch.nn.functional.pad(y_sizes, (0, len(y) - len(y_sizes)), value=0)

        return x, y, ts, x_sizes, y_sizes

    def collater(self, batch):
        x, y, ts, x_sizes, y_sizes = zip(*batch)
        return (
            collate_tensors(x, constant_value=self.vocab.pad_idx),
            collate_tensors(y, constant_value=self.vocab.pad_idx),
            collate_tensors(ts, constant_value=0.0),
            collate_tensors(x_sizes, constant_value=0),
            collate_tensors(y_sizes, constant_value=0),
        )


if __name__ == "__main__":
    from esm.data import Alphabet
    from torch.utils.data import ConcatDataset
    from tqdm import tqdm

    from ..datamodule import PLMRDataModule

    vocab = Vocab.from_esm_alphabet(Alphabet.from_architecture("ESM-1b"))

    data_file1 = "/accounts/projects/yss/stephen.lu/peint/data/dasm/edges/{}/v1jaffePairedCC.txt"
    data_file2 = "/accounts/projects/yss/stephen.lu/peint/data/dasm/edges/{}/v1tangCC.txt"
    data_file3 = (
        "/accounts/projects/yss/stephen.lu/peint/data/dasm/edges/{}/v1vanwinkleheavyTrainCC.txt"
    )

    datasets = {}

    for mode in ["train", "val"]:
        _datasets = []
        for data_file in [data_file1, data_file2, data_file3]:
            dataset = ComplexCherriesDataset(
                data_file=data_file.format(mode),
                min_t=0.0,
                sep_token=".",
                chain_id_offset=1,
            )
            _datasets.append(dataset)

        _dataset = ConcatDataset(_datasets)
        dataset = EncodedLEPTDataset(
            dataset=_dataset,
            vocab=vocab,
            swap_prob=0.5,
            sep_token=".",
        )
        datasets[mode] = dataset

    datamodule = PLMRDataModule(
        dataset=None,
        dataset_train=datasets["train"],
        dataset_val=[datasets["val"]],
        dataset_test=None,
        batch_size=16,
        num_workers=8,
        train_val_split=[0.95, 0.05],
        generator_seed=42,
        pin_memory=False,
        shuffle=True,
    )
    datamodule.setup(stage="fit")

    tr_dataloader = datamodule.train_dataloader()
    val_dataloaders = datamodule.val_dataloader()

    print(f"Dataset size: {len(dataset)}")
    print(f"Train dataset size: {len(datamodule.data_train)}")
    print(f"Number of validation datasets: {len(datamodule.data_val)}")
    print(f"Validation dataset size: {len(datamodule.data_val[0])}")

    print("\n=== Testing Training Dataloader ===")
    for batch in tqdm(iter(tr_dataloader), total=3):
        x, y, ts, x_sizes, y_sizes = batch
        print(f"\nBatch shapes:")
        print(f"  x: {x.shape}")
        print(f"  y: {y.shape}")
        print(f"  ts: {ts.shape}")
        print(f"  x_sizes: {x_sizes.shape}")
        print(f"  y_sizes: {y_sizes.shape}")

        print(f"\nFirst example:")
        print(f"  x tokens: {x[0][:50]}")
        x_decoded = vocab.decode(x[0].numpy())
        print(f"  x decoded: {x_decoded[:100]}")
        print(f"  x_sizes: {x_sizes[0]}")
        print(f"  y_sizes: {y_sizes[0]}")

        # Check separator tokens are present in decoded sequence
        print(f"  Separator token check: '.' found in decoded = {'.' in x_decoded}")
        break

    print("\n=== Testing Validation Dataloader ===")
    for batch in tqdm(iter(val_dataloaders), total=1):
        x, y, ts, x_sizes, y_sizes = batch
        print(f"\nBatch shapes:")
        print(f"  x: {x.shape}")
        print(f"  y: {y.shape}")
        print(f"  ts: {ts.shape}")
        print(f"  x_sizes: {x_sizes.shape}")
        print(f"  y_sizes: {y_sizes.shape}")

        # Check for multi-chain examples
        num_chains = (x_sizes[0] > 0).sum().item()
        print(f"\nFirst example has {num_chains} chain(s)")
        if num_chains > 1:
            x_decoded = vocab.decode(x[0].numpy())
            print(f"  Multi-chain separator check: '.' found = {'.' in x_decoded}")
        break
