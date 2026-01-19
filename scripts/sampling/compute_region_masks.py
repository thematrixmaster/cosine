#!/usr/bin/env python3
"""
compute_region_masks.py

Compute heavy/light chain CDR and FR region boolean masks for a dataset and
save results to disk in a resumable way.

This script mirrors the approach used in the notebook cell (pre-computing
masks). It iterates the provided dataset, decodes amino-acid child sequences,
computes region masks via `create_region_masks`, and saves the lists of
per-sample masks to disk. Progress is saved periodically and on exceptions so
work can be resumed.

Usage (example):
  python scripts/compute_region_masks.py --datapath data/wyatt/subs/edges_joint/nt/d4.txt \
      --out results/masks_d4.pkl --batch-size 64 --num-workers 8 --save-every 200

Notes:
 - The script uses the same dataset and encoder types as the notebook. Run
   it from the repository root so imports resolve correctly.
 - You can set --num-workers to use many CPU cores for the per-sequence mask
   computations (multiprocessing.Pool).
"""
import argparse
import os
import pickle
from multiprocessing import Pool
from pathlib import Path
from typing import Tuple

import numpy as np
from Bio.Seq import Seq

from evo.antibody import create_region_masks
from evo.dataset import ComplexCherriesDataset


def load_complex_dataset(datapath: str):
    """Load ComplexCherriesDataset directly and return it.

    The dataset yields transitions; each item is expected to contain child
    domain sequences (heavy and light) as strings or a list/tuple of strings.
    This function does not wrap the dataset in any DataLoader — we iterate
    directly to keep things simple and resumable.
    """
    if datapath is None:
        raise ValueError("datapath must be provided")
    dataset = ComplexCherriesDataset(data_file=datapath, min_t=0.0, chain_id_offset=1)
    return dataset


def atomic_save(obj, out_path: Path):
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
    os.replace(tmp, out_path)


def compute_masks_for_sequence(seq: str) -> dict:
    """Compute create_region_masks for a single AA sequence.

    Returns a dict of masks (as lists or numpy arrays) so they are picklable.
    """
    masks = create_region_masks(seq)
    # Convert any np.ndarray to list for portability (pickle handles arrays OK,
    # but lists are a little easier to inspect). Keep as np arrays for compactness.
    # We'll convert boolean arrays to lists of ints (0/1) to avoid issues.
    serial = {}
    for k, v in masks.items():
        arr = np.asarray(v, dtype=np.bool_)
        serial[k] = arr.tolist()
    return serial


def maybe_translate(s: str) -> str:
    """Translate nucleotide sequence to amino acid if applicable."""
    s = s.strip()
    if len(s) == 0:
        return s
    # up = set(s.upper())
    # if up.issubset({"A", "T", "G", "C"}) and len(s) % 3 == 0:
    try:
        return str(Seq(s).translate())
    except Exception:
        return s
    # return s


def process_sample(sample_data: Tuple[int, Tuple]) -> Tuple[int, dict, dict]:
    """Process a single sample and return (idx, hv_masks, lt_masks)."""
    idx, sample = sample_data
    try:
        xs, ys, ts, chain_ids = sample
    except Exception:
        return idx, None, None

    # assume heavy is chain 0 and light is chain 1
    hv_seq = ys[0] if len(ys) > 0 else ""
    lt_seq = ys[1] if len(ys) > 1 else ""

    hv_seq = maybe_translate(hv_seq)
    lt_seq = maybe_translate(lt_seq)

    hv_result = compute_masks_for_sequence(hv_seq)
    lt_result = compute_masks_for_sequence(lt_seq)

    return idx, hv_result, lt_result


def main():
    parser = argparse.ArgumentParser(
        description="Compute heavy/light chain region masks for dataset and save resumably."
    )
    parser.add_argument(
        "--datapath",
        type=str,
        required=True,
        help="Path to transitions dataset file (same format as notebook)",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output pickle file to save masks (will be created/resumed)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of worker processes for per-sequence mask computation",
    )
    parser.add_argument(
        "--save-every", type=int, default=500, help="Save intermediate results every N samples"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100,
        help="Number of samples to process in each parallel chunk",
    )
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load or initialize progress
    if out_path.exists():
        print(f"Found existing output {out_path}, resuming...")
        with open(out_path, "rb") as f:
            state = pickle.load(f)
        hv_masks = state.get("hv_masks", [])
        lt_masks = state.get("lt_masks", [])
        processed = int(state.get("processed", 0))
    else:
        hv_masks = []
        lt_masks = []
        processed = 0

    print(f"Processed so far: {processed} samples. Will append to output {out_path}.")
    print(f"Using {args.num_workers} workers for parallel processing.")

    dataset = load_complex_dataset(args.datapath)

    pool = Pool(processes=max(1, args.num_workers))

    saved_at = processed

    try:
        # Use pool.imap for parallel processing with progress tracking
        sample_iterator = ((idx, sample) for idx, sample in enumerate(dataset) if idx >= processed)

        for idx, hv_result, lt_result in pool.imap(
            process_sample, sample_iterator, chunksize=args.chunk_size
        ):
            if hv_result is None or lt_result is None:
                print(f"Warning: unexpected sample format at index {idx}; skipping")
                processed = idx + 1
                continue

            hv_masks.append(hv_result)
            lt_masks.append(lt_result)
            processed = idx + 1

            if processed - saved_at >= args.save_every:
                atomic_save(
                    {"hv_masks": hv_masks, "lt_masks": lt_masks, "processed": processed}, out_path
                )
                print(f"Saved progress at {processed} samples to {out_path}")
                saved_at = processed

            # Update progress bar manually
            if processed % 100 == 0:
                print(f"Processed {processed} samples...")

    except KeyboardInterrupt:
        print("Interrupted by user, saving progress...")
        atomic_save({"hv_masks": hv_masks, "lt_masks": lt_masks, "processed": processed}, out_path)
        pool.terminate()
        pool.join()
        raise
    except Exception as e:
        print(f"Encountered exception: {e!r}. Saving progress up to {processed} samples...")
        atomic_save({"hv_masks": hv_masks, "lt_masks": lt_masks, "processed": processed}, out_path)
        pool.terminate()
        pool.join()
        raise
    finally:
        pool.close()
        pool.join()

    # final save
    atomic_save({"hv_masks": hv_masks, "lt_masks": lt_masks, "processed": processed}, out_path)
    print(f"Finished. Total samples processed: {processed}. Output saved to {out_path}")


if __name__ == "__main__":
    main()
