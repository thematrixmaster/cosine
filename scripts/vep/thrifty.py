from typing import Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
import pandas as pd
import torch
from scipy.special import logsumexp
from tqdm import tqdm

from scripts.vep.base import PairedBaseDMSAnalyzer
from pathlib import Path

PathLike = Union[str, Path]


class ThriftyDMSAnalyzer(PairedBaseDMSAnalyzer):
    """Thrifty SHM neutral model DMS analyzer for paired antibody sequences.

    Scores mutations using the Thrifty nucleotide-level somatic hypermutation
    model. Requires WT nucleotide sequences for the chains being scored.
    Perplexity is computed by marginalizing over synonymous codons at mutated
    positions (logsumexp), so no NT sequences for individual variants are needed.

    Examples:
        >>> thrifty = ThriftyDMSAnalyzer(
        ...     df, heavy_nt_seq=nt_heavy, light_nt_seq=nt_light,
        ...     heavy_chain_col="heavy", light_chain_col="light",
        ... )
        >>> results = thrifty.compute_results(bl_values=[0.01, 0.1, 0.5], chains=["heavy", "light"])
    """

    def __init__(
        self,
        data: Union[PathLike, pd.DataFrame],
        fitness_col: str = "fitness",
        heavy_chain_col: str = "heavy",
        light_chain_col: str = "light",
        heavy_nt_seq: Optional[str] = None,
        light_nt_seq: Optional[str] = None,
        consensus_heavy_aa: Optional[str] = None,
        consensus_light_aa: Optional[str] = None,
        heavy_mut_col: Optional[str] = None,
        light_mut_col: Optional[str] = None,
        neutral_model_name: str = "ThriftyHumV0.2-59",
        multihit_model_name: Optional[str] = "ThriftyHumV0.2-59-hc-tangshm",
    ):
        super().__init__(
            data=data,
            fitness_col=fitness_col,
            heavy_chain_col=heavy_chain_col,
            light_chain_col=light_chain_col,
            consensus_heavy_aa=consensus_heavy_aa,
            consensus_light_aa=consensus_light_aa,
            heavy_mut_col=heavy_mut_col,
            light_mut_col=light_mut_col,
        )
        self.heavy_wt_nt = heavy_nt_seq
        self.light_wt_nt = light_nt_seq
        self.neutral_model_name = neutral_model_name
        self.multihit_model_name = multihit_model_name
        self._neutral_model = None
        self._multihit_model = None
        self.codon_vocab = None

        if heavy_nt_seq is not None or light_nt_seq is not None:
            from Bio.Seq import Seq
            from evo.tokenization import CodonVocab
            self.codon_vocab = CodonVocab.from_codons()

            if heavy_nt_seq is not None:
                translated = str(Seq(heavy_nt_seq).translate())
                if translated != self.heavy_wt_aa:
                    raise ValueError(
                        f"Heavy NT translation mismatch.\n"
                        f"Expected: {self.heavy_wt_aa}\nGot: {translated}"
                    )
            if light_nt_seq is not None:
                translated = str(Seq(light_nt_seq).translate())
                if translated != self.light_wt_aa:
                    raise ValueError(
                        f"Light NT translation mismatch.\n"
                        f"Expected: {self.light_wt_aa}\nGot: {translated}"
                    )

    def compute_results(
        self,
        bl_values: Union[float, List[float]],
        chains: Union[str, List[str]] = ["heavy", "light"],
        show_progress: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """Score variants with Thrifty neutral model.

        Scoring is always single-chain for "heavy"/"light" (only the focal
        chain's LL is computed) and always both-chain for "combined" (mutations
        applied to both chains simultaneously).

        | chain      | What is scored                    |
        |------------|-----------------------------------|
        | "heavy"    | heavy LL (mut) only               |
        | "light"    | light LL (mut) only               |
        | "combined" | heavy LL (mut) + light LL (mut)   |

        Args:
            bl_values: Thrifty branch length(s).
            chains: Which chains to score. Subset of ["heavy", "light", "combined"].
            show_progress: Show tqdm progress bars.

        Returns:
            Dict mapping chain name to DataFrame with ll_{bl} and ppl_{bl} columns added.
        """
        if isinstance(bl_values, (int, float)):
            bl_values = [bl_values]
        if isinstance(chains, str):
            chains = [chains]

        for chain in chains:
            if chain in ("heavy", "combined") and self.heavy_wt_nt is None:
                raise ValueError("heavy_nt_seq required for Thrifty scoring of heavy chain.")
            if chain in ("light", "combined") and self.light_wt_nt is None:
                raise ValueError("light_nt_seq required for Thrifty scoring of light chain.")

        if self.codon_vocab is None:
            from evo.tokenization import CodonVocab
            self.codon_vocab = CodonVocab.from_codons()

        results = {}

        for chain in chains:
            if chain == "combined":
                df = self.df_all.copy()
                df = df.assign(
                    mut=df["heavy_mut"].fillna("") + "|" + df["light_mut"].fillna("")
                ).set_index("mut")
                seq_len = len(self.heavy_wt_aa) + len(self.light_wt_aa)
            else:
                df = (self.df_heavy if chain == "heavy" else self.df_light).copy()
                seq_len = len(self.heavy_wt_aa) if chain == "heavy" else len(self.light_wt_aa)

            for bl in bl_values:
                col_suffix = str(bl)

                if chain == "combined":
                    print(f"Computing neutral probabilities for heavy chain at bl={bl:.6g}...")
                    heavy_probs = self._compute_neutral_probs(self.heavy_wt_nt, bl)
                    print(f"Computing neutral probabilities for light chain at bl={bl:.6g}...")
                    light_probs = self._compute_neutral_probs(self.light_wt_nt, bl)
                    for mut in tqdm(
                        df.index, disable=not show_progress, desc=f"Thrifty combined bl={bl}"
                    ):
                        heavy_mut_str, light_mut_str = mut.split("|", 1)
                        ll = self._compute_combined_neutral_ll(
                            heavy_mut_str, light_mut_str, heavy_probs, light_probs
                        )
                        df.loc[mut, f"ll_{col_suffix}"] = ll
                        df.loc[mut, f"ppl_{col_suffix}"] = np.exp(-ll / seq_len)
                else:
                    wt_nt = self.heavy_wt_nt if chain == "heavy" else self.light_wt_nt
                    print(f"Computing neutral probabilities for {chain} chain at bl={bl:.6g}...")
                    probs = self._compute_neutral_probs(wt_nt, bl)
                    for mut in tqdm(
                        df.index, disable=not show_progress, desc=f"Thrifty {chain} bl={bl}"
                    ):
                        ll = self._single_chain_ll(wt_nt, probs, self._parse_muts(mut))
                        df.loc[mut, f"ll_{col_suffix}"] = ll
                        df.loc[mut, f"ppl_{col_suffix}"] = np.exp(-ll / seq_len)

            results[chain] = df

        return results

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    def _load_thrifty_models(self):
        if self._neutral_model is not None:
            return
        try:
            from netam import pretrained
            # print(f"Loading Thrifty neutral model: {self.neutral_model_name}")
            self._neutral_model = pretrained.load(self.neutral_model_name)
            if self.multihit_model_name:
                try:
                    # print(f"Loading multihit model: {self.multihit_model_name}")
                    self._multihit_model = pretrained.load_multihit(
                        self.multihit_model_name, device="cpu"
                    )
                except Exception as e:
                    warnings.warn(
                        f"Failed to load multihit model: {e}. "
                        "Continuing with base model only."
                    )
                    self._multihit_model = None
        except ImportError:
            raise ImportError(
                "netam not installed. Install with: pip install netam\n"
                "Required for Thrifty neutral mutation correction."
            )

    def _compute_neutral_probs(self, wt_nt_seq: str, branch_length: float) -> torch.Tensor:
        """Compute per-codon log probabilities using the Thrifty neutral model.

        Returns:
            Tensor of shape (num_codons, 64) with log probabilities.
            Codon index: i*16 + j*4 + k where (i,j,k) are ACGT indices of each base.
        """
        from netam.molevol import neutral_codon_probs_of_seq
        self._load_thrifty_models()

        scaled_rates, csp_logits = self._neutral_model([wt_nt_seq])
        seq_len = len(wt_nt_seq)
        scaled_rates = scaled_rates[0, :seq_len]
        csp_logits = csp_logits[0, :seq_len]
        csps = torch.softmax(csp_logits, dim=-1)
        mask = torch.ones(seq_len, dtype=torch.bool)

        codon_probs = neutral_codon_probs_of_seq(
            wt_nt_seq, mask, scaled_rates, csps, branch_length,
            multihit_model=self._multihit_model,
        )
        return torch.log(codon_probs + 1e-10)

    def _compute_combined_neutral_ll(
        self,
        heavy_mut_str: str,
        light_mut_str: str,
        heavy_probs: torch.Tensor,
        light_probs: torch.Tensor,
    ) -> float:
        """LL over both chains with mutations applied to both simultaneously."""
        heavy_ll = self._single_chain_ll(
            self.heavy_wt_nt, heavy_probs, self._parse_muts(heavy_mut_str)
        )
        light_ll = self._single_chain_ll(
            self.light_wt_nt, light_probs, self._parse_muts(light_mut_str)
        )
        return heavy_ll + light_ll

    def _single_chain_ll(
        self,
        wt_nt: str,
        probs: torch.Tensor,
        mutations: Dict[int, str],
    ) -> float:
        """Per-sequence LL for one chain, marginalizing over synonymous codons.

        For each codon position:
          - If mutated: ll += logsumexp over all codons encoding the new AA
          - If WT: ll += log prob of the WT codon

        Args:
            wt_nt: WT nucleotide sequence for this chain.
            probs: Log-prob tensor of shape (num_codons, 64).
            mutations: Dict mapping 0-indexed codon position to new amino acid.
        """
        nt_to_idx = {"A": 0, "C": 1, "G": 2, "T": 3}

        def codon_to_idx(c):
            return nt_to_idx[c[0]] * 16 + nt_to_idx[c[1]] * 4 + nt_to_idx[c[2]]

        wt_codon_idxs = [codon_to_idx(wt_nt[i:i + 3]) for i in range(0, len(wt_nt), 3)]
        ll = 0.0
        for pos, wt_idx in enumerate(wt_codon_idxs):
            if pos in mutations:
                new_aa = mutations[pos]
                codons = [c for c, aa in self.codon_vocab.GENETIC_CODE.items() if aa == new_aa]
                ll += logsumexp([probs[pos, codon_to_idx(c)].item() for c in codons])
            else:
                ll += probs[pos, wt_idx].item()
        return ll

    @staticmethod
    def _parse_muts(mut_str: str) -> Dict[int, str]:
        """Parse "A23V" or "A23V,T45G" into {23: 'V', 45: 'G'} (0-indexed positions)."""
        if not mut_str or mut_str == "WT":
            return {}
        parts = mut_str.split(",") if "," in mut_str else [mut_str]
        return {int(m[1:-1]): m[-1] for m in parts}
