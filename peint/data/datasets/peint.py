import torch
from evo.dataset import CherriesDataset, EncodedCherriesDataset, TorchWrapperDataset
from evo.tensor import collate_tensors, mask_tensor
from evo.tokenization import Vocab


class EncodedPEINTDataset(EncodedCherriesDataset):
    def __init__(
        self,
        dataset: CherriesDataset,
        vocab: Vocab,
        mask_prob: float = 0.15,
        random_token_prob: float = 0.1,
        leave_unmasked_prob: float = 0.1,
        *args,
        **kwargs,
    ):
        super().__init__(dataset, vocab, *args, **kwargs)
        self._mask_prob = mask_prob
        self._random_token_prob = random_token_prob
        self._leave_unmasked_prob = leave_unmasked_prob

    def __getitem__(self, index):
        x, y, t = super().__getitem__(index)
        x_src, x_tgt = mask_tensor(
            x,
            self.vocab,  # x_tgt gets pad tok at masked pos
            mask_prob=self._mask_prob,
            random_token_prob=self._random_token_prob,
            leave_unmasked_prob=self._leave_unmasked_prob,
        )
        y_src, y_tgt = y[:-1], y[1:]  # Shift y for auto-regressive training
        x_src_pad_mask = x_src == self.vocab.pad_idx
        y_src_pad_mask = y_src == self.vocab.pad_idx
        return x_src, x_tgt, y_src, y_tgt, t, x_src_pad_mask, y_src_pad_mask

    def collater(self, batch):
        x_src, x_tgt, y_src, y_tgt, t, x_src_pad_mask, y_src_pad_mask = zip(*batch)
        return (
            collate_tensors(x_src, constant_value=self.vocab.pad_idx),
            collate_tensors(x_tgt, constant_value=self.vocab.pad_idx),
            collate_tensors(y_src, constant_value=self.vocab.pad_idx),
            collate_tensors(y_tgt, constant_value=self.vocab.pad_idx),
            torch.tensor(t, dtype=torch.float32).reshape(-1, 1),
            collate_tensors(x_src_pad_mask, constant_value=True, dtype=torch.bool),
            collate_tensors(y_src_pad_mask, constant_value=True, dtype=torch.bool),
        )


class EncodedAntiCherriesDataset(TorchWrapperDataset):
    def __init__(
        self, dataset: CherriesDataset, vocab: Vocab, sep_token: str = ".", *args, **kwargs
    ):
        super().__init__(dataset=dataset, vocab=vocab, *args, **kwargs)
        self.sep_token = sep_token
        assert self.sep_token in self.vocab.tokens, f"{self.sep_token} not in vocab"

    def __getitem__(self, index: int) -> torch.Tensor:
        x, y, t = super().__getitem__(index)
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


class EncodedAntiPEINTDataset(EncodedAntiCherriesDataset):
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

    def __getitem__(self, index: int) -> torch.Tensor:
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


# Usage example
if __name__ == "__main__":
    from esm.data import Alphabet
    from torch.utils.data import ConcatDataset, DataLoader

    vocab = Vocab.from_esm_alphabet(Alphabet.from_architecture("ESM-1b"))

    data_file1 = (
        "/accounts/projects/yss/stephen.lu/protevo/plmr/data/wyatt/all/edges_joint/new_d1.txt"
    )
    data_file2 = (
        "/accounts/projects/yss/stephen.lu/protevo/plmr/data/wyatt/all/edges_joint/new_d2.txt"
    )

    dataset1 = CherriesDataset(
        data_file=data_file1,
        cache_indices=False,
        min_t=5e-3,
    )

    dataset2 = CherriesDataset(
        data_file=data_file2,
        cache_indices=False,
        min_t=5e-3,
    )

    _dataset = ConcatDataset([dataset1, dataset2])

    dataset = EncodedAntiPEINTDataset(
        dataset=_dataset,
        vocab=vocab,
        sep_token=".",
        mask_prob=0.15,
        random_token_prob=0.0,
        leave_unmasked_prob=0.0,
    )

    dataloader = DataLoader(dataset=dataset, batch_size=8, collate_fn=dataset.collater)

    item = dataset[0]

    batch = next(iter(dataloader))
    x_src, x_tgt, y_src, y_tgt, t, x_sizes, y_sizes = batch

    # breakpoint()
