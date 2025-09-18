# Metrics for scoring generated proteins

import os
import shutil
import subprocess
import sys
import tempfile
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm

warnings.filterwarnings("ignore")

from peint.data.utils import Protein
from peint.metrics.api import GenerativeMetric

# =============================================================================
# SEQUENCE IDENTITY METRICS (using MMseqs2)
# =============================================================================


class SequenceIdentityMetric(GenerativeMetric):
    """Simple MMseqs2-based sequence identity metric"""

    def __init__(
        self,
        reference_path: str,
        sensitivity: float = 7.5,
        evalue: float = 0.001,
        min_seq_id: float = 0.3,
        coverage: float = 0.8,
        max_seqs: int = 1000,
        threads: int = 8,
        additional_args: Optional[List[str]] = None,
    ):
        """
        Args:
            reference_path: Path to reference FASTA file OR MMseqs2 database
                           For databases: path to the database prefix (e.g. "uniref50_db")
                           For FASTA: path to .fasta/.fa file
            sensitivity: Search sensitivity (1.0-8.0+). Higher = more sensitive but slower.
                        Default 7.5 is good for protein similarity search.
            evalue: E-value threshold. Lower = more stringent. Default 0.001.
            min_seq_id: Minimum sequence identity (0.0-1.0). Default 0.3 (30%).
            coverage: Query coverage requirement (0.0-1.0). Default 0.8 (80%).
            max_seqs: Maximum number of target sequences per query. Default 1000.
            threads: Number of threads to use. Default 8.
            additional_args: Additional MMseqs2 arguments as list of strings.
        """
        super().__init__("sequence_identity")
        self.reference_path = reference_path
        self.sensitivity = sensitivity
        self.evalue = evalue
        self.min_seq_id = min_seq_id
        self.coverage = coverage
        self.max_seqs = max_seqs
        self.threads = threads
        self.additional_args = additional_args or []

        self._check_mmseqs2()
        self._setup_reference()

    def _check_mmseqs2(self):
        """Check if mmseqs2 is available"""
        try:
            subprocess.run(["mmseqs", "version"], capture_output=False, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError(
                "mmseqs2 not found. Install from: https://github.com/soedinglab/MMseqs2"
            )

    def _setup_reference(self):
        """Setup reference database or FASTA file"""
        if self.reference_path.endswith((".fasta", ".fa", ".fas")):
            if not os.path.exists(self.reference_path):
                raise FileNotFoundError(f"Reference FASTA not found: {self.reference_path}")
            self.is_database = False
        elif self.reference_path.endswith(".parquet"):
            if not os.path.exists(self.reference_path):
                raise FileNotFoundError(f"Reference Parquet file not found: {self.reference_path}")
            # Convert Parquet to temporary FASTA
            table = pq.read_table(self.reference_path)
            sequences = table.column("sequence").to_pylist()
            if not sequences:
                raise ValueError("Parquet file does not contain any sequences.")
            tmp_fasta_file = tempfile.NamedTemporaryFile(delete=False, suffix=".fasta")
            with open(tmp_fasta_file.name, "w") as f:
                for i, seq in enumerate(sequences):
                    f.write(f">seq_{i}\n{seq}\n")
            self.reference_path = tmp_fasta_file.name
            self.is_database = False
        else:
            # Check for MMseqs2 database files
            if not os.path.exists(f"{self.reference_path}.dbtype"):
                raise FileNotFoundError(f"MMseqs2 database not found: {self.reference_path}")
            self.is_database = True

    def _write_fasta(self, proteins: List[Protein], output_path: str):
        """Write proteins to FASTA file"""
        with open(output_path, "w") as f:
            for i, protein in enumerate(proteins):
                name = f"seq_{i}"
                f.write(f">{name}\n{protein}\n")

    def compute(self, samples: List[Protein]) -> Dict[str, List[float]]:
        """
        Compute sequence identity using MMseqs2 easy-search
        """
        if not samples:
            return []

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Write query sequences
            query_fasta = os.path.join(tmp_dir, "query.fasta")
            self._write_fasta(samples, query_fasta)

            # Build MMseqs2 command
            results_file = os.path.join(tmp_dir, "results.m8")
            cmd = [
                "mmseqs",
                "easy-search",
                query_fasta,
                self.reference_path,
                results_file,
                tmp_dir,
                "--threads",
                str(self.threads),
                "-s",
                str(self.sensitivity),
                "-e",
                str(self.evalue),
                "--min-seq-id",
                str(self.min_seq_id),
                "-c",
                str(self.coverage),
                "--max-seqs",
                str(self.max_seqs),
            ]
            cmd.extend(self.additional_args)
            try:
                subprocess.run(cmd, check=True, capture_output=False, text=True)
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"MMseqs2 failed: {e.stderr}")

            return self._parse_results(results_file, len(samples))

    def aggregate(self, metrics):
        return {
            "max_identity": np.mean(metrics["max_identity"]),
            "num_hits": np.mean(metrics["num_hits"]),
        }

    def _parse_results(self, results_file: str, num_samples: int) -> Dict[str, List[float]]:
        """Parse MMseqs2 output and return per-sample metrics"""
        identities = {}
        if os.path.exists(results_file):
            with open(results_file, "r") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    query_id = parts[0]
                    identity = float(parts[2])  # % identity is column 3

                    if query_id not in identities:
                        identities[query_id] = []
                    identities[query_id].append(identity)

        # Return per-sample metrics
        per_sample_metrics = defaultdict(list)
        for i in range(num_samples):
            query_id = f"seq_{i}"
            if query_id in identities:
                sample_identities = identities[query_id]
                per_sample_metrics["max_identity"].append(max(sample_identities))
                per_sample_metrics["num_hits"].append(len(sample_identities))
            else:
                per_sample_metrics["max_identity"].append(0.0)
                per_sample_metrics["num_hits"].append(0)
        return per_sample_metrics


# =============================================================================
# FOLDED STRUCTURE METRICS (Using ColabFold and Foldseek)
# =============================================================================


class StructureFoldingMetric(GenerativeMetric):
    """Folds sequences using ColabFold or ESMFold and computes structure metrics"""

    def __init__(
        self,
        reference_path: Optional[str] = None,
        folding_method: str = "colabfold",
    ):
        """
        Args:
            reference_path: Path to reference PDBs/database or None
            folding_method: "colabfold" or "esmfold"
        """
        super().__init__("structure_folding")
        self.reference_path = reference_path
        self.folding_method = folding_method
        self._check_dependencies()

    def _check_dependencies(self):
        """Check if required tools are available."""
        try:
            if self.folding_method == "colabfold":
                cmd = f"colabfold_batch --help"
                subprocess.run(cmd, shell=True, capture_output=True, check=True)
            elif self.folding_method == "esmfold":
                pass
        except (subprocess.CalledProcessError, FileNotFoundError, ImportError):
            raise RuntimeError(f"{self.folding_method} dependencies not found.")

    def compute(self, samples: List[Protein]) -> Dict[str, List[float]]:
        """Compute structure metrics for protein samples."""
        if not samples:
            return []

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Batch fold all proteins
            folded_pdbs = self._batch_fold_proteins(samples, tmp_path)

            # Batch compute TM-scores if references provided
            tm_scores_dict = {}
            if self.reference_path:
                tm_scores_dict = self._batch_compute_tm_scores(folded_pdbs, tmp_path)

            # Compute individual metrics
            results = defaultdict(list)
            for i, (sequence, pdb_file) in enumerate(zip(samples, folded_pdbs)):
                if pdb_file:
                    plddt_scores = self._extract_plddt(pdb_file)
                    plddt_metrics = self._get_plddt_metrics(plddt_scores)

                    sample_id = f"seq_{i}"
                    tm_metrics = {"max_tm_score": tm_scores_dict.get(sample_id, 0.0)}

                    quality_metrics = self._get_quality_metrics(pdb_file)

                    for key, value in plddt_metrics.items():
                        results[key].append(value)
                    for key, value in tm_metrics.items():
                        results[key].append(value)
                    for key, value in quality_metrics.items():
                        results[key].append(value)
                else:
                    default_metrics = self._default_metrics()
                    for key, value in default_metrics.items():
                        results[key].append(value)

        return results

    def aggregate(self, metrics):
        """Aggregate metrics across samples."""
        aggregated = {}
        for key, values in metrics.items():
            if values:
                aggregated[key] = np.mean(values)
            else:
                aggregated[key] = 0.0
        return aggregated

    def _batch_fold_proteins(self, samples: List[Protein], tmp_path: Path) -> List[Optional[str]]:
        """Batch fold all proteins using selected method."""
        if self.folding_method == "colabfold":
            return self._batch_fold_colabfold(samples, tmp_path)
        elif self.folding_method == "esmfold":
            return self._batch_fold_esmfold(samples, tmp_path)

    def _batch_fold_colabfold(self, samples: List[Protein], tmp_path: Path) -> List[Optional[str]]:
        """Batch fold using ColabFold."""
        fasta_file = tmp_path / "all_sequences.fasta"
        with open(fasta_file, "w") as f:
            for i, sequence in enumerate(samples):
                f.write(f">seq_{i}\n{sequence}\n")

        cmd = f"colabfold_batch {fasta_file} {tmp_path} --num-models 1 --num-recycle 1"
        try:
            print(f"Running ColabFold command: {cmd}")
            subprocess.run(
                cmd,
                shell=True,
                check=True,
                stdout=sys.stdout,  # Direct output to console
                stderr=sys.stderr,  # Direct errors to console
                text=True,  # Handle as text
                bufsize=0,  # Unbuffered
            )
        except subprocess.CalledProcessError:
            raise

        folded_pdbs = []
        for i in range(len(samples)):
            pdb_files = list(tmp_path.glob(f"seq_{i}_*.pdb"))
            folded_pdbs.append(str(pdb_files[0]) if pdb_files else None)

        return folded_pdbs

    def _batch_fold_esmfold(self, samples: List[Protein], tmp_path: Path) -> List[Optional[str]]:
        """Batch fold using ESMFold via torch.hub."""
        import torch

        model = torch.hub.load("facebookresearch/esm:main", "esmfold_v1")
        model = model.eval()
        if torch.cuda.is_available():
            model = model.cuda()

        folded_pdbs = []
        for i, sequence in tqdm(enumerate(samples)):
            try:
                with torch.no_grad():
                    pdb_string = model.infer_pdb(sequence)

                pdb_file = tmp_path / f"seq_{i}.pdb"
                with open(pdb_file, "w") as f:
                    f.write(pdb_string)

                folded_pdbs.append(str(pdb_file))
            except Exception:
                folded_pdbs.append(None)

        return folded_pdbs

    def _batch_compute_tm_scores(
        self, folded_pdbs: List[Optional[str]], tmp_path: Path
    ) -> Dict[str, float]:
        """Batch compute TM-scores using foldseek."""
        valid_pdbs = [(i, pdb) for i, pdb in enumerate(folded_pdbs) if pdb]
        if not valid_pdbs:
            return {}

        query_dir = tmp_path / "query_structures"
        query_dir.mkdir()

        for i, pdb_path in valid_pdbs:
            seq_id = f"seq_{i}"
            shutil.copy(pdb_path, query_dir / f"{seq_id}.pdb")

        try:
            result_file = tmp_path / "tm_results.tsv"
            subprocess.run(
                [
                    "foldseek",
                    "easy-search",
                    str(query_dir),
                    self.reference_path,
                    str(result_file),
                    str(tmp_path),
                    "--format-output",
                    "query,target,alntmscore",
                ],
                check=True,
                capture_output=False,
            )

            tm_scores = {}
            if result_file.exists():
                with open(result_file, "r") as f:
                    for line in f:
                        try:
                            parts = line.strip().split("\t")
                            query_id = parts[0].replace(".pdb", "")
                            tm_score = float(parts[2])
                            tm_scores[query_id] = max(tm_scores.get(query_id, 0.0), tm_score)
                        except (ValueError, IndexError):
                            continue
            return tm_scores
        except subprocess.CalledProcessError:
            raise

    def _extract_plddt(self, pdb_file: str) -> List[float]:
        """Extract plDDT scores from B-factor column."""
        scores = []
        with open(pdb_file, "r") as f:
            for line in f:
                if line.startswith("ATOM") and len(line) > 66:
                    try:
                        scores.append(float(line[60:66].strip()))
                    except ValueError:
                        continue
        return scores

    def _get_plddt_metrics(self, scores: List[float]) -> Dict[str, float]:
        """Compute plDDT-based metrics."""
        if not scores:
            return {"avg_plddt": 0.0, "min_plddt": 0.0, "confident_residues": 0.0}

        scores_array = np.array(scores)
        return {
            "avg_plddt": float(np.mean(scores_array)),
            "min_plddt": float(np.min(scores_array)),
            "confident_residues": float(np.sum(scores_array > 70) / len(scores_array)),
        }

    def _get_quality_metrics(self, pdb_file: str) -> Dict[str, float]:
        """Compute basic structural quality metrics."""
        coords = self._parse_coordinates(pdb_file)

        if len(coords) < 2:
            return {"num_clashes": 0, "radius_of_gyration": 0.0}

        coords_array = np.array(coords)

        # Count clashes (atoms < 2.0 Å apart)
        distances = np.sqrt(
            np.sum((coords_array[:, np.newaxis] - coords_array[np.newaxis, :]) ** 2, axis=2)
        )
        clashes = np.sum((distances > 0) & (distances < 2.0)) // 2

        # Radius of gyration
        center = np.mean(coords_array, axis=0)
        rog = np.sqrt(np.mean(np.sum((coords_array - center) ** 2, axis=1)))

        return {"num_clashes": int(clashes), "radius_of_gyration": float(rog)}

    def _parse_coordinates(self, pdb_file: str) -> List[tuple]:
        """Parse atomic coordinates from PDB."""
        coords = []
        with open(pdb_file, "r") as f:
            for line in f:
                if line.startswith("ATOM") and len(line) > 54:
                    try:
                        x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                        coords.append((x, y, z))
                    except ValueError:
                        continue
        return coords

    def _default_metrics(self) -> Dict[str, Any]:
        """Return default metrics when folding fails."""
        return {
            "avg_plddt": 0.0,
            "min_plddt": 0.0,
            "confident_residues": 0.0,
            "max_tm_score": 0.0,
            "num_clashes": 0,
            "radius_of_gyration": 0.0,
        }


# Example usage
if __name__ == "__main__":
    proteins = [
        Protein(
            "MTSENPLLALREKISALDEKLLALLAERRELAGGVGKAKLLPVRDIDRERDLLERLITLGKAAHLDAHYITRLFQLIIEDSVLTQQALLQQHLNKINPHSARIAFLGPKGSYSHLAARQYAARHFEQFIESGCAKFADIFNQVETGQADYAVVPIENTSSGAINDVYD"
        ),
        Protein(
            "LLQHTSLSIVGEMTLTIDHCLLVSGTTDLSTINTVYSHPQPFQQCSKFLNRYPHWKIEYTESTSAAMEKVAQAKSPHVAALGSEAGGTLYGLQVLERIEANQRQNFTRFVVLARKAINVSDQVPAKTTLLMATGQQAGALVEALLVLRNHSLIMTRLESRPIHGNPWEEMFYLDIQANLESAEMQKALKELGEITRSMKVLGCYPSENVVPVDPT"
        ),
    ]

    if 0:
        seqidentity = SequenceIdentityMetric("data/sets/b1lpa6_ecosm_russ_2020.parquet")
        print("Computing sequence identity...")
        results = seqidentity.compute(proteins)
        print("Sequence Identity Results:", results)
        print("Aggregated Results:", seqidentity.aggregate(results))

    if 1:
        strfolded = StructureFoldingMetric("data/external/1ECM.cif")
        print("Computing structure folding metrics...")
        results = strfolded.compute(proteins)
        print("Structure Folding Results:", results)
        print("Aggregated Results:", strfolded.aggregate(results))
