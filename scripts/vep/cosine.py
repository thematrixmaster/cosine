from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import os
import tempfile
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

from scripts.vep.base import PairedBaseDMSAnalyzer

PathLike = Union[str, Path]


class CosineDMSAnalyzer(PairedBaseDMSAnalyzer):
    """CTMC-based (CoSiNE) DMS analyzer for paired antibody sequences.

    Scores single-amino-acid or combined heavy+light chain mutations using a
    trained CTMCModule. Supports paired context (heavy.light fed together) or
    single-chain scoring.

    Examples:
        >>> cosine = CosineDMSAnalyzer(df, ctmc_module, heavy_chain_col="heavy", light_chain_col="light")
        >>> results = cosine.compute_results(t_values=[1, 5, 20], chains=["heavy", "light"], paired=True)
        >>> corr = cosine.compute_correlations(results, metric_cols=["ll_5.0"])
    """

    def __init__(
        self,
        data: Union[PathLike, pd.DataFrame],
        ctmc_module,
        fitness_col: str = "fitness",
        heavy_chain_col: str = "heavy",
        light_chain_col: str = "light",
        consensus_heavy_aa: Optional[str] = None,
        consensus_light_aa: Optional[str] = None,
        heavy_mut_col: Optional[str] = None,
        light_mut_col: Optional[str] = None,
        device: str = "cuda",
        batch_size: int = 32,
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
        self.ctmc_module = ctmc_module
        self.device = device
        self.batch_size = batch_size

    def compute_results(
        self,
        t_values: Union[float, List[float]] = 5.0,
        chains: Union[str, List[str]] = ["heavy", "light"],
        paired: bool = True,
        show_progress: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """Score variants with CTMC model.

        Two scoring axes:
          - chain: which variants to score ("heavy", "light", or "combined")
          - paired: whether to provide full heavy.light context to the model

        | chain      | paired | Context fed to model              |
        |------------|--------|-----------------------------------|
        | "heavy"    | True   | WT_H.WT_L → MUT_H.WT_L  t        |
        | "heavy"    | False  | WT_H → MUT_H  t                   |
        | "light"    | True   | WT_H.WT_L → WT_H.MUT_L  t        |
        | "light"    | False  | WT_L → MUT_L  t                   |
        | "combined" | N/A    | WT_H.WT_L → MUT_H.MUT_L  t       |

        Args:
            t_values: Branch length(s) to evaluate.
            chains: Which chains to score. Subset of ["heavy", "light", "combined"].
            paired: If True, score mutations in full heavy.light context.
                   If False, score single chain in isolation. Ignored for "combined".
            show_progress: Show tqdm progress bars.

        Returns:
            Dict mapping chain name to DataFrame with ll_{t} and ppl_{t} columns added.
            ppl is computed as exp(-ll / seq_len), where seq_len is the total length
            of the sequence(s) fed to the model.
        """
        if isinstance(t_values, (int, float)):
            t_values = [t_values]
        if isinstance(chains, str):
            chains = [chains]

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
                seq_len = (
                    len(self.heavy_wt_aa) + len(self.light_wt_aa)
                    if paired
                    else (len(self.heavy_wt_aa) if chain == "heavy" else len(self.light_wt_aa))
                )

            for t_value in t_values:
                col_suffix = str(t_value)
                transitions, mutation_labels = self._build_ctmc_transitions(chain, t_value, paired)
                if not transitions:
                    warnings.warn(f"No mutations found for chain='{chain}', skipping t={t_value}.")
                    continue

                desc = f"CTMC {chain} (t={t_value})"
                lls, _ = self._run_ctmc_inference(transitions, show_progress, desc)

                for mut, ll in zip(mutation_labels, lls):
                    df.loc[mut, f"ll_{col_suffix}"] = ll
                    df.loc[mut, f"ppl_{col_suffix}"] = np.exp(-ll / seq_len)

            results[chain] = df

        return results

    def _build_ctmc_transitions(
        self,
        chain: str,
        t_value: float,
        paired: bool = True,
    ) -> Tuple[List[str], List[str]]:
        """Build AA transition strings for CTMC inference.

        Returns:
            (transitions, mutation_labels): parallel lists of transition strings
            and their corresponding mutation identifiers.
        """
        df = self.df_all if chain == "combined" else (
            self.df_heavy if chain == "heavy" else self.df_light
        )

        transitions = []
        mutation_labels = []

        for i, row in df.iterrows():
            if chain == "combined":
                mut = f"{row['heavy_mut']}|{row['light_mut']}"
                transition = (
                    f"{self.heavy_wt_aa}.{self.light_wt_aa} "
                    f"{row[self.heavy_chain_col]}.{row[self.light_chain_col]} "
                    f"{t_value}"
                )
            elif paired:
                mut = row.name
                if chain == "heavy":
                    transition = (
                        f"{self.heavy_wt_aa}.{self.light_wt_aa} "
                        f"{row[self.heavy_chain_col]}.{self.light_wt_aa} "
                        f"{t_value}"
                    )
                else:
                    transition = (
                        f"{self.heavy_wt_aa}.{self.light_wt_aa} "
                        f"{self.heavy_wt_aa}.{row[self.light_chain_col]} "
                        f"{t_value}"
                    )
            else:
                mut = row.name
                if chain == "heavy":
                    transition = f"{self.heavy_wt_aa} {row[self.heavy_chain_col]} {t_value}"
                else:
                    transition = f"{self.light_wt_aa} {row[self.light_chain_col]} {t_value}"

            transitions.append(transition)
            mutation_labels.append(mut)

        return transitions, mutation_labels

    def _run_ctmc_inference(
        self,
        transitions: List[str],
        show_progress: bool = True,
        desc: str = "CTMC inference",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run CTMC inference on a list of transition strings.

        Returns:
            (lls, ppls): per-sequence log-likelihoods and perplexities.
        """
        from evo.dataset import ComplexCherriesDataset
        from peint.data.datasets.ctmc import CTMCDataset
        from peint.data.datamodule import PLMRDataModule

        datafile = tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w")
        with open(datafile.name, "w") as f:
            f.write(f"{len(transitions)} transitions\n")
            f.write("\n".join(transitions))

        dataset = ComplexCherriesDataset(
            data_file=datafile.name, min_t=0.0, chain_id_offset=1
        )
        ctmc_dataset = CTMCDataset(
            dataset=dataset,
            sep_token=".",
            vocab=self.ctmc_module.net.vocab,
        )
        dataloader = PLMRDataModule(
            dataset=ctmc_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
        )._dataloader_template(dataset=ctmc_dataset, training=False)

        vocab = self.ctmc_module.net.vocab
        special_tok_idxs = torch.tensor(
            [
                vocab.bos_idx, vocab.pad_idx, vocab.eos_idx, vocab.unk_idx, vocab.mask_idx,
                vocab.tokens_to_idx.get("<null_1>", -1),
                vocab.tokens_to_idx.get(".", -1),
                vocab.tokens_to_idx.get("X", -1),
                vocab.tokens_to_idx.get("B", -1),
                vocab.tokens_to_idx.get("Z", -1),
                vocab.tokens_to_idx.get("O", -1),
                vocab.tokens_to_idx.get("U", -1),
            ],
            device=self.device,
        )

        lls, ppls = [], []
        autocast_device = "cuda" if "cuda" in self.device else "cpu"
        for batch in tqdm(dataloader, disable=not show_progress, desc=desc):
            batch = [b.to(self.device) for b in batch]
            x, y, t, x_sizes = batch

            with torch.no_grad(), torch.autocast(device_type=autocast_device, dtype=torch.bfloat16):
                Q, pi = self.ctmc_module.net(x, x_sizes=x_sizes)
                P = self.ctmc_module.net.exp_Qt(Q, t)
                log_probs = self.ctmc_module.net.log_Px(P, x)

            nll = F.cross_entropy(
                log_probs.transpose(-1, -2),
                y,
                ignore_index=vocab.pad_idx,
                reduction="none",
            )
            aa_mask = torch.isin(y, special_tok_idxs, invert=True)
            ll = (-nll * aa_mask.float()).sum(dim=-1)
            lls.append(ll.cpu().numpy())

            nll_mean = (nll * aa_mask.float()).sum(dim=-1) / aa_mask.float().sum(dim=-1).clamp(min=1)
            ppls.append(torch.exp(nll_mean).cpu().numpy())

        os.unlink(datafile.name)

        if not lls:
            return np.array([]), np.array([])
        return np.concatenate(lls), np.concatenate(ppls)

    def selection_score(
        self,
        cosine_results: Dict[str, pd.DataFrame],
        thrifty_results: Dict[str, pd.DataFrame],
        log_transform_fitness: bool = False,
        correlation_method: str = "spearman",
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
        """Subtract Thrifty SHM LLs from CTMC LLs for all (t, bl) combinations.

        Implements the LL subtraction logic from DMSAnalyzer.plot_correlation_heatmap
        without producing a plot. Returns both the corrected LL columns and a
        per-chain correlation matrix for all (cosine_t × thrifty_bl) combinations.

        Args:
            cosine_results: Output of compute_results().
            thrifty_results: Output of ThriftyDMSAnalyzer.compute_results().
            log_transform_fitness: Log-transform fitness before correlating.
            correlation_method: "pearson" or "spearman".

        Returns:
            corrected_results: Dict[str, pd.DataFrame] with corrected_ll_{t}_{bl}
                columns added to each chain's DataFrame.
            correlation_dfs: Dict[str, pd.DataFrame] keyed by chain, each indexed by
                ["no_correction", "bl=X", ...] and columns ["t=X", ...],
                where values are Pearson/Spearman r between corrected LL and fitness.
        """
        corr_func = pearsonr if correlation_method == "pearson" else spearmanr

        shared_chains = [c for c in cosine_results if c in thrifty_results]
        if not shared_chains:
            raise ValueError("No shared chains between cosine_results and thrifty_results.")

        def _parse_suffix(col):
            s = col.split("_", 1)[1]
            return np.inf if s == "inf" else float(s)

        corrected_results = {}
        correlation_dfs = {}

        for chain in shared_chains:
            cf = cosine_results[chain].copy()
            tf = thrifty_results[chain].copy()

            cosine_ll_cols = sorted(c for c in cf.columns if c.startswith("ll_"))
            thrifty_ll_cols = sorted(c for c in tf.columns if c.startswith("ll_"))

            cosine_t_values = [_parse_suffix(c) for c in cosine_ll_cols]
            thrifty_bl_values = [_parse_suffix(c) for c in thrifty_ll_cols]

            shared_idx = cf.index.intersection(tf.index)
            merged = cf.loc[shared_idx].merge(
                tf.loc[shared_idx],
                left_index=True,
                right_index=True,
                how="inner",
                suffixes=("_cosine", "_thrifty"),
            )

            fitness_col_merged = (
                self.fitness_col + "_cosine"
                if self.fitness_col + "_cosine" in merged.columns
                else self.fitness_col
            )

            def _fitness(sub):
                f = sub[fitness_col_merged]
                return np.log(f) if log_transform_fitness else f

            n_rows = 1 + len(thrifty_ll_cols)
            corr_matrix = np.full((n_rows, len(cosine_ll_cols)), np.nan)

            # Row 0: no correction (raw CTMC LL)
            for j, cosine_col in enumerate(cosine_ll_cols):
                cc = cosine_col + "_cosine" if cosine_col + "_cosine" in merged.columns else cosine_col
                sub = merged[[cc, fitness_col_merged]].dropna()
                if len(sub) >= 2:
                    corr, _ = corr_func(sub[cc], _fitness(sub))
                    corr_matrix[0, j] = corr

            # Rows 1+: corrected LL = cosine_ll - thrifty_ll
            for i, thrifty_col in enumerate(thrifty_ll_cols):
                for j, cosine_col in enumerate(cosine_ll_cols):
                    cc = cosine_col + "_cosine" if cosine_col + "_cosine" in merged.columns else cosine_col
                    tc = thrifty_col + "_thrifty" if thrifty_col + "_thrifty" in merged.columns else thrifty_col
                    sub = merged[[cc, tc, fitness_col_merged]].dropna()
                    if len(sub) >= 2:
                        corrected = sub[cc] - sub[tc]
                        corr, _ = corr_func(corrected, _fitness(sub))
                        corr_matrix[i + 1, j] = corr

            row_labels = ["no_correction"] + [f"bl={bl}" for bl in thrifty_bl_values]
            col_labels = [f"t={t}" if t != np.inf else "t=inf" for t in cosine_t_values]
            correlation_dfs[chain] = pd.DataFrame(corr_matrix, index=row_labels, columns=col_labels)

            # Add corrected_ll_{t}_{bl} columns
            for thrifty_col in thrifty_ll_cols:
                if thrifty_col not in tf.columns:
                    continue
                bl_str = thrifty_col.split("_", 1)[1]
                for cosine_col in cosine_ll_cols:
                    if cosine_col not in cf.columns:
                        continue
                    t_str = cosine_col.split("_", 1)[1]
                    cf.loc[shared_idx, f"corrected_ll_{t_str}_{bl_str}"] = (
                        cf.loc[shared_idx, cosine_col] - tf.loc[shared_idx, thrifty_col]
                    )
            corrected_results[chain] = cf

        return corrected_results, correlation_dfs
