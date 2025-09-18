import numpy as np
import torch

from evo.dataset import CherriesDataset, TorchWrapperDataset
from evo.tensor import collate_tensors, mask_tensor
from evo.tokenization import Vocab


class EncodedPIPETCherriesDataset(TorchWrapperDataset):
    def __init__(
        self, dataset: CherriesDataset, vocab: Vocab, sep_token: str = ".", *args, **kwargs
    ):
        super().__init__(dataset=dataset, vocab=vocab, *args, **kwargs)
        self.sep_token = sep_token
        assert self.sep_token in self.vocab.tokens, f"{self.sep_token} not in vocab"

    def __getitem__(self, index: int):
        x, y, t = super().__getitem__(index)

        if self.sep_token not in x:
            print(f"Found bad data at index {index}: {x}")
            raise ValueError()

        if self.sep_token not in y:
            print(f"Found bad data at index {index}: {y}")
            raise ValueError()

        x_heavy, x_light = x.split(self.sep_token)
        y_heavy, y_light = y.split(self.sep_token)

        x_heavy = torch.from_numpy(self.vocab.encode_single_sequence(x_heavy))
        x_light = torch.from_numpy(self.vocab.encode_single_sequence(x_light))
        x = torch.cat([x_heavy, x_light], dim=0)  # <cls> x1 <eos> <cls> x2 <eos>
        x_sizes = torch.zeros_like(x, dtype=torch.long)
        x_sizes[0], x_sizes[1] = x_heavy.size(0), x_light.size(0)

        y = torch.from_numpy(self.vocab.encode_single_sequence(y))  # <cls> y1 <sep> y2 <eos>
        y_sizes = torch.zeros_like(y, dtype=torch.long)
        y_sizes[0], y_sizes[1] = len(y_heavy) + 1, len(y_light) + 1

        ts = torch.zeros_like(y, dtype=torch.float32)
        ts[0], ts[1] = t, t
        return x, y, ts, x_sizes, y_sizes

    def collater(self, batch):
        xs, ys, ts, x_sizes, y_sizes = zip(*batch)
        xs = collate_tensors(xs, constant_value=self.vocab.pad_idx)
        ys = collate_tensors(ys, constant_value=self.vocab.pad_idx)
        ts = collate_tensors(ts, constant_value=0.0)
        x_sizes = collate_tensors(x_sizes, constant_value=0)
        y_sizes = collate_tensors(y_sizes, constant_value=0)
        return xs, ys, ts, x_sizes, y_sizes


class EncodedPIPETDataset(EncodedPIPETCherriesDataset):
    def __init__(
        self,
        dataset: CherriesDataset,
        vocab: Vocab,
        mask_prob: float = 0.15,
        random_token_prob: float = 0.1,
        leave_unmasked_prob: float = 0.1,
        sep_token: str = ".",
        *args,
        **kwargs,
    ):
        super().__init__(dataset, vocab, sep_token=sep_token, *args, **kwargs)
        self._mask_prob = mask_prob
        self._random_token_prob = random_token_prob
        self._leave_unmasked_prob = leave_unmasked_prob

    def __getitem__(self, index: int):
        x, y, t, x_sizes, y_sizes = super().__getitem__(index)
        x_src, x_tgt = mask_tensor(
            x,
            self.vocab,  # x_tgt gets pad tok at masked pos
            mask_prob=self._mask_prob,
            random_token_prob=self._random_token_prob,
            leave_unmasked_prob=self._leave_unmasked_prob,
            extra_special_tok_idx=[self.vocab.tokens_to_idx[self.sep_token]],
        )
        t = t[:-1]
        y_src, y_tgt = y[:-1], y[1:]  # Shift y for auto-regressive training
        attn_mask_x_sizes = x_sizes
        attn_mask_y_sizes = y_sizes[:-1]
        return x_src, x_tgt, y_src, y_tgt, t, attn_mask_x_sizes, attn_mask_y_sizes

    def collater(self, batch):
        x_src, x_tgt, y_src, y_tgt, t, x_sizes, y_sizes = zip(*batch)
        return (
            collate_tensors(x_src, constant_value=self.vocab.pad_idx),
            collate_tensors(x_tgt, constant_value=self.vocab.pad_idx),
            collate_tensors(y_src, constant_value=self.vocab.pad_idx),
            collate_tensors(y_tgt, constant_value=self.vocab.pad_idx),
            collate_tensors(t, constant_value=0.0),
            collate_tensors(x_sizes, constant_value=0),
            collate_tensors(y_sizes, constant_value=0),
        )


class EncodedDMSDataset(EncodedPIPETCherriesDataset):
    """DMS dataset for PIPET that processes protein pairs for deep mutational scanning."""

    def __init__(self, t: float = 7.0, sep_token: str = ".", *args, **kwargs):
        super().__init__(sep_token=sep_token, *args, **kwargs)
        self.t = t

    def __getitem__(self, index: int):
        x, y, mut_str = self.dataset.__getitem__(index)
        x_heavy, x_light = x.split(self.sep_token)
        y_heavy, y_light = y.split(self.sep_token)

        x_heavy = torch.from_numpy(self.vocab.encode_single_sequence(x_heavy))
        x_light = torch.from_numpy(self.vocab.encode_single_sequence(x_light))
        x = torch.cat([x_heavy, x_light], dim=0)  # <cls> x1 <eos> <cls> x2 <eos>
        x_sizes = torch.zeros_like(x, dtype=torch.long)
        x_sizes[0], x_sizes[1] = x_heavy.size(0), x_light.size(0)

        y = torch.from_numpy(self.vocab.encode_single_sequence(y))  # <cls> y1 <sep> y2 <eos>
        y_sizes = torch.zeros_like(y, dtype=torch.long)
        y_sizes[0], y_sizes[1] = len(y_heavy) + 1, len(y_light) + 1

        ts = torch.zeros_like(y, dtype=torch.float32)
        ts[0], ts[1] = self.t, self.t

        ts = ts[:-1]
        y_src, y_tgt = y[:-1], y[1:]  # Shift y for nll calculation
        attn_mask_x_sizes = x_sizes
        attn_mask_y_sizes = y_sizes[:-1]

        return x, y_src, y_tgt, mut_str, ts, attn_mask_x_sizes, attn_mask_y_sizes

    def collater(self, batch):
        x, y_src, y_tgt, mut_str, t, x_sizes, y_sizes = zip(*batch)
        return (
            collate_tensors(x, constant_value=self.vocab.pad_idx),
            collate_tensors(y_src, constant_value=self.vocab.pad_idx),
            collate_tensors(y_tgt, constant_value=self.vocab.pad_idx),
            np.array(mut_str, dtype=object),
            collate_tensors(t, constant_value=0.0),
            collate_tensors(x_sizes, constant_value=0),
            collate_tensors(y_sizes, constant_value=0),
        )


# Usage example
if __name__ == "__main__":
    from esm.data import Alphabet
    from torch.utils.data import ConcatDataset
    from tqdm import tqdm

    from ..datamodule import PLMRDataModule

    vocab = Vocab.from_esm_alphabet(Alphabet.from_architecture("ESM-1b"))

    data_file1 = "/accounts/projects/yss/stephen.lu/peint/data/wyatt/all/edges_joint_pipet/d1.txt"
    data_file2 = "/accounts/projects/yss/stephen.lu/peint/data/wyatt/all/edges_joint_pipet/d2.txt"
    data_file3 = "/accounts/projects/yss/stephen.lu/peint/data/wyatt/all/edges_joint_pipet/d3.txt"
    data_file4 = "/accounts/projects/yss/stephen.lu/peint/data/wyatt/all/edges_joint_pipet/d4.txt"

    datasets = []
    for data_file in [data_file1, data_file2, data_file3, data_file4]:
        dataset = CherriesDataset(
            data_file=data_file,
            cache_indices=False,
            min_t=5e-3,
        )
        datasets.append(dataset)

    _dataset = ConcatDataset(datasets)

    dataset = EncodedPIPETDataset(
        dataset=_dataset,
        vocab=vocab,
        sep_token=".",
        mask_prob=0.15,
        random_token_prob=0.0,
        leave_unmasked_prob=0.0,
    )

    datamodule = PLMRDataModule(
        dataset=dataset,
        dataset_train=None,
        dataset_val=None,
        dataset_test=None,
        batch_size=32,
        num_workers=8,
        train_val_split=[0.95, 0.05],
        generator_seed=42,
        pin_memory=False,
    )

    datamodule.setup(stage="fit")

    tr_dataloader = datamodule.train_dataloader()
    val_dataloader = datamodule.val_dataloader()

    print(f"Dataset size: {len(dataset)}")
    print(f"Train dataset size: {len(datamodule.data_train)}")

    print(f"Number of validation datasets: {len(datamodule.data_val)}")
    print(f"Validation dataset size: {len(datamodule.data_val[0])}")

    # dataloader = DataLoader(
    #     dataset=dataset,
    #     batch_size=32,
    #     collate_fn=dataset.collater,
    #     num_workers=8,
    #     shuffle=False,
    # )

    # for batch in tqdm(iter(dataloader)):
    #     pass

    for batch in tqdm(iter(tr_dataloader)):
        pass

    for batch in tqdm(iter(val_dataloader)):
        pass

    # item = dataset[0]

    # batch = next(iter(dataloader))
    # x_src, x_tgt, y_src, y_tgt, t, x_sizes, y_sizes = batch

    # breakpoint()
