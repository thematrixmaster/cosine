import torch
from evo.dataset import TorchWrapperDataset, ComplexCherriesDataset
from evo.tensor import collate_tensors
from evo.tokenization import Vocab

from peint.models.frameworks.dfm import ConditionalPath
from peint.models.frameworks.editflows import (
    Alignment,
    rm_gap_tokens,
    make_ut_mask_from_z,
)


class EditFlowsDataset(TorchWrapperDataset):
    def __init__(
        self,
        vocab: Vocab,
        dataset: ComplexCherriesDataset,
        conditional_path: ConditionalPath,
        alignment: Alignment,
        sep_token: str = ".",
        random_cherry_order: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(dataset=dataset, vocab=vocab, *args, **kwargs)
        self.alignment = alignment
        self.conditional_path = conditional_path
        self.x_vocab_size = len(vocab)
        self.z_vocab_size = self.x_vocab_size + 1  # +1 for gap token
        self.pad_token_id = vocab.pad_idx
        self.gap_token_id = len(vocab)
        self.sep_token = sep_token
        self.random_cherry_order = random_cherry_order

    def __getitem__(self, index):
        xs, ys, ts, _ = super().__getitem__(index)

        # if random order is true, then permute order of x and y randomly
        if self.random_cherry_order:
            perm = torch.randperm(len(xs))
            xs = [xs[i] for i in perm]
            ys = [ys[i] for i in perm]

        # x is always embedded together as one sequence with a separator token
        x_sizes = torch.tensor([len(x) + 1 for x in xs], dtype=torch.long)
        x_sizes[0] += self.vocab.prepend_bos  # add bos to first chain
        x_sizes[-1] += (
            self.vocab.append_eos - 1
        )  # add eos to last chain and subtract 1 for sep token
        xs = torch.from_numpy(self.vocab.encode_single_sequence(self.sep_token.join(xs)))

        # y is always embedded together as one sequence for autoregressive training
        ys = torch.from_numpy(self.vocab.encode_single_sequence(self.sep_token.join(ys)))
        t = torch.tensor([ts[0]], dtype=torch.float32)

        # Apply padding to the right of x_sizes to match x_sizes
        x_sizes = torch.nn.functional.pad(x_sizes, (0, len(xs) - len(x_sizes)), value=0)

        x0, x1 = xs.unsqueeze(0), ys.unsqueeze(0)
        x0, x1, z0, z1 = self.alignment(x0, x1)
        flow_t = torch.rand(x0.size(0)).reshape(-1, 1)
        zt = self.conditional_path.sample(z0, z1, flow_t, vocab_size=self.z_vocab_size)
        xt, x_pad_mask, z_gap_mask, z_pad_mask = rm_gap_tokens(
            zt, self.gap_token_id, self.pad_token_id
        )
        uz_mask = make_ut_mask_from_z(
            zt, z1, self.x_vocab_size, self.pad_token_id, self.gap_token_id
        )
        return (
            xt.squeeze(0),
            x_sizes, t,
            x_pad_mask.squeeze(0),
            z_gap_mask.squeeze(0),
            z_pad_mask.squeeze(0),
            uz_mask.squeeze(0),
        )

    def collater(self, batch):
        xt, x_sizes, t, x_pad_mask, z_gap_mask, z_pad_mask, uz_mask = zip(*batch)
        return (
            collate_tensors(xt, constant_value=self.pad_token_id),
            collate_tensors(x_sizes, constant_value=0),
            collate_tensors(t, constant_value=0.0),
            collate_tensors(x_pad_mask, constant_value=False, dtype=torch.bool),
            collate_tensors(z_gap_mask, constant_value=False, dtype=torch.bool),
            collate_tensors(z_pad_mask, constant_value=False, dtype=torch.bool),
            collate_tensors(uz_mask, constant_value=False, dtype=torch.bool),
        )


# Example usage
if __name__ == "__main__":
    from esm.data import Alphabet
    from evo.dataset import ComplexCherriesDataset
    from evo.tokenization import Vocab

    from peint.models.frameworks.dfm import CubicScheduler, FactorizedPath
    from peint.models.frameworks.editflows import OptimalAlignment, AllOptimalAlignments

    vocab = Vocab.from_esm_alphabet(Alphabet.from_architecture("ESM-1b"))
    dataset = ComplexCherriesDataset(
        data_file="/accounts/projects/yss/stephen.lu/peint/data/wyatt/subs/edges_joint/aa/d1.txt",
        min_t=0.0,
        sep_token=".",
    )
    conditional_path = FactorizedPath(kappa=CubicScheduler(), vocab_size=len(vocab)+1)
    alignment = AllOptimalAlignments(strategy="uniform", vocab=vocab, add_bos_token=True, add_eos_token=True)
    editflows_dataset = EditFlowsDataset(
        vocab=vocab,
        dataset=dataset,
        conditional_path=conditional_path,
        alignment=alignment,
        sep_token=".",
        random_cherry_order=False,
    )
    xt, x_sizes, t, x_pad_mask, z_gap_mask, z_pad_mask, uz_mask = editflows_dataset[0]
    print(xt.shape, x_sizes.shape, t.shape, x_pad_mask.shape, z_gap_mask.shape, z_pad_mask.shape, uz_mask.shape)
    breakpoint()
    