from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

from evo.dms import get_site_by_site_consensus
from scripts.vep.utils import get_mutations

PathLike = Union[str, Path]
ResultsLike = Union[pd.DataFrame, Dict[str, pd.DataFrame]]


class BaseDMSAnalyzer:
    """
    Base class for single-chain DMS analysis.

    Handles data loading, consensus resolution, and mutation detection.
    All work is done on self.df_variants which has mutation strings as index.
    Subclasses implement compute_results() with a specific scoring model.

    Examples:
        >>> analyzer = DummyDMSAnalyzer("data/dms.csv", fitness_col="fitness", mut_col="mutations")
        >>> results = analyzer.compute_results()
        >>> analyzer.compute_correlations(results, metric_cols=["fitness_pred"])
    """

    def __init__(
        self,
        data: Union[PathLike, pd.DataFrame],
        fitness_col: str = "fitness",
        seq_col: Optional[str] = None,
        consensus_seq: Optional[str] = None,
        consensus_row_idx: Optional[int] = None,
        mut_col: Optional[str] = None,
    ):
        """
        Args:
            data: CSV path or DataFrame.
            fitness_col: Column containing fitness values.
            seq_col: Column containing AA sequences (optional).
            consensus_seq: Explicit WT/consensus AA sequence.
            consensus_row_idx: Use this row's sequence as consensus.
                Mutually exclusive with consensus_seq.
                If neither given, computes site-by-site consensus from seq_col.
            mut_col: Pre-computed mutation column (skips auto-detection).
                Mutations should be formatted as e.g. "A23V" or "A23V,T45G".
        """
        # Load data
        if isinstance(data, (str, Path)):
            df_full = pd.read_csv(data)
        else:
            df_full = data.copy()

        self.fitness_col = fitness_col
        self.seq_col = seq_col

        # Validate required columns
        if fitness_col not in df_full.columns:
            raise ValueError(f"Fitness column '{fitness_col}' not found in data")
        if seq_col is not None and seq_col not in df_full.columns:
            raise ValueError(f"Sequence column '{seq_col}' not found in data")

        if consensus_seq is not None and consensus_row_idx is not None:
            raise ValueError("Provide at most one of consensus_seq or consensus_row_idx")

        # Resolve consensus
        if consensus_seq is not None:
            self.consensus = consensus_seq
        elif consensus_row_idx is not None:
            if seq_col is None:
                raise ValueError("seq_col required to extract consensus from row_idx")
            self.consensus = df_full.iloc[consensus_row_idx][seq_col]
            # print(f"Using row {consensus_row_idx} as consensus")
        else:
            if seq_col is None:
                raise ValueError("consensus_seq, consensus_row_idx, or seq_col required to determine consensus")
            self.consensus = get_site_by_site_consensus(df_full, seq_col)
            # print("Computed site-by-site consensus")

        # Detect mutations
        df_copy = df_full.copy()
        if mut_col is not None:
            if mut_col not in df_full.columns:
                raise ValueError(f"Mutation column '{mut_col}' not found in data")
            df_copy["_mut"] = df_copy[mut_col]
            # print(f"Using provided mutation column: '{mut_col}'")
        elif seq_col is not None:
            df_copy["_mut"] = df_copy[seq_col].apply(
                lambda x: get_mutations(x, self.consensus)
            )
            # print("Computing mutations from sequences")
        else:
            raise ValueError("mut_col or seq_col required for mutation detection")

        # Filter to non-WT variants and set mut as index
        is_variant = (
            df_copy["_mut"].notna()
            & (df_copy["_mut"] != "")
            & (df_copy["_mut"] != "WT")
        )
        self.df_variants = (
            df_copy[is_variant]
            .copy()
            .rename(columns={"_mut": "mut"})
            .set_index("mut")
        )
        print(f"Found {len(self.df_variants)} non-WT variants")

    def compute_results(self, **kwargs) -> pd.DataFrame:
        """Score all variants. Returns df_variants with added prediction columns."""
        raise NotImplementedError("Subclasses must implement compute_results()")

    def compute_correlations(
        self,
        results: pd.DataFrame,
        metric_cols: Optional[List[str]] = None,
        log_transform_fitness: bool = False,
        correlation_method: str = "spearman",
    ) -> pd.DataFrame:
        """
        Correlate prediction columns against fitness.

        Args:
            results: DataFrame output of compute_results().
            metric_cols: Columns to correlate. If None, uses all numeric columns
                except fitness_col.
            log_transform_fitness: Log-transform fitness before correlating.
            correlation_method: "pearson" or "spearman".

        Returns:
            DataFrame with columns [metric, correlation, p_value, n_samples].
        """
        corr_func = pearsonr if correlation_method == "pearson" else spearmanr

        fitness = results[self.fitness_col]
        if log_transform_fitness:
            fitness = np.log(fitness)

        if metric_cols is None:
            metric_cols = [
                c for c in results.columns
                if c != self.fitness_col and pd.api.types.is_numeric_dtype(results[c])
            ]

        rows = []
        for col in metric_cols:
            preds = results[col]
            mask = ~(preds.isna() | fitness.isna())
            p, f = preds[mask], fitness[mask]
            if len(p) < 2:
                warnings.warn(f"Fewer than 2 valid samples for '{col}', skipping")
                continue
            corr, pval = corr_func(p, f)
            rows.append({"metric": col, "correlation": corr, "p_value": pval, "n_samples": int(mask.sum())})

        return pd.DataFrame(rows)

    def plot_scatter_t_values(
        self,
        results: pd.DataFrame,
        t_values: List[float],
        negate: bool = True,
        log_transform_fitness: bool = False,
        figsize_per_panel: Tuple[float, float] = (4, 4),
        title: Optional[str] = None,
    ) -> Tuple[plt.Figure, np.ndarray]:
        """Scatter plots of fitness vs negative perplexity for multiple t values, side by side.

        Args:
            results: DataFrame output of compute_results().
            t_values: List of t values to plot (must have corresponding ppl_{t} columns).
            negate: If True, plot negative perplexity (so higher = better).
            log_transform_fitness: Log-transform fitness before plotting.
            figsize_per_panel: (width, height) per subplot panel.

        Returns:
            (fig, axes) tuple.
        """
        n = len(t_values)
        fig, axes = plt.subplots(
            1, n,
            figsize=(figsize_per_panel[0] * n, figsize_per_panel[1]),
            sharey=True,
        )
        if n == 1:
            axes = [axes]

        fitness = results[self.fitness_col].copy()
        if log_transform_fitness:
            fitness = np.log(fitness)
        ylabel = "Log fitness" if log_transform_fitness else "Fitness"

        for ax, t in zip(axes, t_values):
            col = f"ppl_{t}"
            if col not in results.columns:
                ax.set_visible(False)
                continue

            preds = results[col].copy()
            if negate:
                preds = -preds
            mask = ~(preds.isna() | fitness.isna())

            corr, _ = spearmanr(preds[mask], fitness[mask])

            ax.scatter(
                preds[mask], fitness[mask],
                alpha=0.4, edgecolor="black", linewidths=0.3, s=18,
            )
            ax.set_xlabel(f"{'Neg. ' if negate else ''}Perplexity  (t={t})", fontsize=11)
            ax.set_title(fr"Spearman $\rho$ = {corr:.3f}", fontsize=11)
            ax.grid(True, alpha=0.25, linewidth=0.6)
            ax.spines[["top", "right"]].set_visible(False)

        axes[0].set_ylabel(ylabel, fontsize=11)
        fig.tight_layout()
        if title is not None:
            fig.suptitle(title, fontsize=12)
            fig.subplots_adjust(top=0.85)
        return fig, np.array(axes)

    def plot_scatter(
        self,
        results: pd.DataFrame,
        metric_col: str,
        metric_col_name: Optional[str] = None,
        negate: bool = True,
        log_transform_fitness: bool = False,
        figsize: Tuple[float, float] = (6, 5),
        title: Optional[str] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Scatter plot of a prediction column vs fitness."""
        fitness = results[self.fitness_col].copy()
        if log_transform_fitness:
            fitness = np.log(fitness)
        preds = results[metric_col].copy()
        mask = ~(preds.isna() | fitness.isna())
        if negate:
            preds = -preds
        corr, pval = spearmanr(preds[mask], fitness[mask])

        if metric_col_name is None:
            metric_col_name = f"Negative Perplexity, t={metric_col.split('_')[-1]}"


        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(preds[mask], fitness[mask], alpha=0.5, edgecolor="black", linewidths=0.5)
        ax.set_xlabel(metric_col_name)
        ax.set_ylabel("Log fitness" if log_transform_fitness else "Fitness")
        ax.set_title(fr"Spearman $\rho$={corr:.3f}, (n={mask.sum()})")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if title is not None:
            plt.suptitle(title, fontsize=12)
            plt.subplots_adjust(top=0.85)
        return fig, ax


class PairedBaseDMSAnalyzer(BaseDMSAnalyzer):
    """Base class for paired heavy+light chain antibody DMS analysis.

    Handles data loading, consensus resolution, and mutation detection for both chains.
    Subclasses implement compute_results() with a specific scoring model.

    Attributes:
        df_heavy: Heavy-chain variant DataFrame, index = mutation string (e.g. "A105V").
        df_light: Light-chain variant DataFrame, index = mutation string.
        df_all: All variants with any mutation (union of heavy/light), used for combined mode.
        heavy_wt_aa: WT heavy-chain amino acid sequence.
        light_wt_aa: WT light-chain amino acid sequence.
    """

    def __init__(
        self,
        data: Union[PathLike, pd.DataFrame],
        fitness_col: str = "fitness",
        heavy_chain_col: str = "heavy",
        light_chain_col: str = "light",
        consensus_heavy_aa: Optional[str] = None,
        consensus_light_aa: Optional[str] = None,
        heavy_mut_col: Optional[str] = None,
        light_mut_col: Optional[str] = None,
    ):
        """
        Args:
            data: CSV path or DataFrame.
            fitness_col: Column containing fitness values.
            heavy_chain_col: Column containing heavy-chain AA sequences.
            light_chain_col: Column containing light-chain AA sequences.
            consensus_heavy_aa: Explicit WT heavy AA sequence. If None, computed
                site-by-site from heavy_chain_col.
            consensus_light_aa: Explicit WT light AA sequence. If None, computed
                site-by-site from light_chain_col.
            heavy_mut_col: Pre-computed heavy mutation column (skips auto-detection).
            light_mut_col: Pre-computed light mutation column (skips auto-detection).
        """
        # Set minimal parent attributes so inherited utility methods work.
        # We do NOT call super().__init__() — the single-chain init is incompatible.
        self.fitness_col = fitness_col
        self.seq_col = None
        self.consensus = None
        self.df_variants = None

        if isinstance(data, (str, Path)):
            df_full = pd.read_csv(data)
        else:
            df_full = data.copy()

        required = [heavy_chain_col, light_chain_col, fitness_col]
        missing = [c for c in required if c not in df_full.columns]
        if missing:
            raise ValueError(f"Data missing required columns: {missing}")

        # Resolve consensus sequences
        if consensus_heavy_aa is not None:
            self.heavy_wt_aa = consensus_heavy_aa
        else:
            self.heavy_wt_aa = get_site_by_site_consensus(df_full, heavy_chain_col)
            # print("Computed heavy chain site-by-site consensus")

        if consensus_light_aa is not None:
            self.light_wt_aa = consensus_light_aa
        else:
            self.light_wt_aa = get_site_by_site_consensus(df_full, light_chain_col)
            # print("Computed light chain site-by-site consensus")

        # Detect mutations
        df_copy = df_full.copy()
        if heavy_mut_col is not None:
            if heavy_mut_col not in df_full.columns:
                raise ValueError(f"Heavy mutation column '{heavy_mut_col}' not found in data")
            df_copy["heavy_mut"] = df_copy[heavy_mut_col]
            print(f"Using provided heavy mutation column: '{heavy_mut_col}'")
        else:
            df_copy["heavy_mut"] = df_copy[heavy_chain_col].apply(
                lambda x: get_mutations(x, self.heavy_wt_aa)
            )
            # print("Computing heavy chain mutations from sequences")

        if light_mut_col is not None:
            if light_mut_col not in df_full.columns:
                raise ValueError(f"Light mutation column '{light_mut_col}' not found in data")
            df_copy["light_mut"] = df_copy[light_mut_col]
            print(f"Using provided light mutation column: '{light_mut_col}'")
        else:
            df_copy["light_mut"] = df_copy[light_chain_col].apply(
                lambda x: get_mutations(x, self.light_wt_aa)
            )
            # print("Computing light chain mutations from sequences")

        def _is_variant(col):
            return df_copy[col].notna() & (df_copy[col] != "") & (df_copy[col] != "WT")

        heavy_has_mut = _is_variant("heavy_mut")
        light_has_mut = _is_variant("light_mut")

        self.df_heavy = (
            df_copy[heavy_has_mut]
            .copy()
            .rename(columns={"heavy_mut": "mut"})
            .set_index("mut")
        )
        self.df_light = (
            df_copy[light_has_mut]
            .copy()
            .rename(columns={"light_mut": "mut"})
            .set_index("mut")
        )
        # df_all keeps both mutation columns (used by combined mode)
        self.df_all = df_copy[heavy_has_mut | light_has_mut].copy()

        self.heavy_chain_col = heavy_chain_col
        self.light_chain_col = light_chain_col

        print(
            f"Found {len(self.df_heavy)} heavy chain mutations "
            f"and {len(self.df_light)} light chain mutations"
        )

    def compute_results(self, **kwargs) -> Dict[str, pd.DataFrame]:
        raise NotImplementedError("Subclasses must implement compute_results()")

    def compute_correlations(
        self,
        results: ResultsLike,
        metric_cols: Optional[List[str]] = None,
        log_transform_fitness: bool = False,
        correlation_method: str = "spearman",
        chains: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Correlate prediction columns against fitness.

        Args:
            results: DataFrame or dict of DataFrames from compute_results().
            metric_cols: Columns to correlate. If None, uses all numeric columns
                except fitness_col.
            log_transform_fitness: Log-transform fitness before correlating.
            correlation_method: "pearson" or "spearman".
            chains: Which chains to include (for dict results). None = all.

        Returns:
            DataFrame with columns [chain, metric, correlation, p_value, n_samples].
        """
        if isinstance(results, pd.DataFrame):
            rows = self._correlate_df(
                results, metric_cols, log_transform_fitness, correlation_method
            )
            return pd.DataFrame(rows)

        selected = {
            k: v for k, v in results.items()
            if chains is None or k in chains
        }
        all_rows = []
        for chain, df in selected.items():
            for row in self._correlate_df(df, metric_cols, log_transform_fitness, correlation_method):
                row["chain"] = chain
                all_rows.append(row)

        cols = ["chain", "metric", "correlation", "p_value", "n_samples"]
        out = pd.DataFrame(all_rows)
        return out[[c for c in cols if c in out.columns]]

    def _correlate_df(
        self,
        df: pd.DataFrame,
        metric_cols: Optional[List[str]],
        log_transform_fitness: bool,
        correlation_method: str,
    ) -> List[Dict]:
        corr_func = pearsonr if correlation_method == "pearson" else spearmanr
        fitness = df[self.fitness_col].copy()
        if log_transform_fitness:
            fitness = np.log(fitness)
        if metric_cols is None:
            metric_cols = [
                c for c in df.columns
                if c != self.fitness_col and pd.api.types.is_numeric_dtype(df[c])
            ]
        rows = []
        for col in metric_cols:
            preds = df[col]
            mask = ~(preds.isna() | fitness.isna())
            p, f = preds[mask], fitness[mask]
            if len(p) < 2:
                warnings.warn(f"Fewer than 2 valid samples for '{col}', skipping")
                continue
            corr, pval = corr_func(p, f)
            rows.append({"metric": col, "correlation": corr, "p_value": pval, "n_samples": int(mask.sum())})
        return rows
