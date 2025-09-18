import numpy as np
import torch

from evo.dataset import CherriesDataset, EncodedCherriesDataset
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


class EncodedDMSDataset(EncodedCherriesDataset):
    def __init__(self, t: float = 7.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.t = t

    def __getitem__(self, index):
        x, y, mut_str = super().__getitem__(
            self, index
        )  # we use the t slot to store fitness and use a constant t value instead
        y_src, y_tgt = y[:-1], y[1:]  # Shift y for nll calculation
        x_pad_mask = x == self.vocab.pad_idx
        y_src_pad_mask = y_src == self.vocab.pad_idx
        return x, y_src, y_tgt, mut_str, self.t, x_pad_mask, y_src_pad_mask

    def collater(self, batch):
        x, y_src, y_tgt, mut_str, t, x_pad_mask, y_src_pad_mask = zip(*batch)
        return (
            collate_tensors(x, constant_value=self.vocab.pad_idx),
            collate_tensors(y_src, constant_value=self.vocab.pad_idx),
            collate_tensors(y_tgt, constant_value=self.vocab.pad_idx),
            np.array(mut_str, dtype=object),
            torch.tensor(t, dtype=torch.float32).reshape(-1, 1),
            collate_tensors(x_pad_mask, constant_value=True, dtype=torch.bool),
            collate_tensors(y_src_pad_mask, constant_value=True, dtype=torch.bool),
        )
