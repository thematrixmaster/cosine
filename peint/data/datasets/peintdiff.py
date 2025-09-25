import numpy as np
import torch

from evo.dataset import (
    CherriesDataset,
    FastaDataset,
    TorchWrapperDataset,
    WeightedConcatDataset,
)
from evo.tensor import collate_tensors
from evo.tokenization import Vocab


class EncodedPEINTDiffDataset(TorchWrapperDataset):
    def __init__(
        self,
        dataset: CherriesDataset | FastaDataset,
        vocab: Vocab,
        *args,
        **kwargs,
    ):
        super().__init__(dataset=dataset, vocab=vocab, *args, **kwargs)

    @property
    def weights(self):
        if isinstance(self.dataset, WeightedConcatDataset):
            return self.dataset.weights
        return np.ones(len(self.dataset), dtype=np.float32)

    def __getitem__(self, index):
        item = super().__getitem__(index)
        if isinstance(item, tuple):
            x, y, t = item
            x = torch.from_numpy(self.vocab.encode_single_sequence(x))
            y = torch.from_numpy(self.vocab.encode_single_sequence(y))
            labeled = True
            y_src, y_tgt = y[:-1], y[1:]
            y_src_pad_mask = y_src == self.vocab.pad_idx
        else:
            x = item
            x = torch.from_numpy(self.vocab.encode_single_sequence(x))
            labeled = False
            y_src, y_tgt, t = None, None, None
            y_src_pad_mask = None
        x_pad_mask = x == self.vocab.pad_idx
        return x, x_pad_mask, labeled, y_src, y_tgt, t, y_src_pad_mask

    def collater(self, batch):
        x, x_pad_mask, labeled, y_src, y_tgt, t, y_src_pad_mask = zip(*batch)
        t = [t for t in t if t is not None]
        return (
            # Encoder input
            collate_tensors(x, constant_value=self.vocab.pad_idx),
            collate_tensors(x_pad_mask, constant_value=True, dtype=torch.bool),
            # Decoder input
            torch.tensor(labeled, dtype=torch.bool),
            collate_tensors(y_src, constant_value=self.vocab.pad_idx),
            collate_tensors(y_tgt, constant_value=self.vocab.pad_idx),
            torch.tensor(t, dtype=torch.float32).reshape(-1, 1),
            collate_tensors(y_src_pad_mask, constant_value=True, dtype=torch.bool),
        )

    def sampler(self):
        return torch.utils.data.WeightedRandomSampler(
            weights=self.weights,
            num_samples=len(self.dataset),
        )
