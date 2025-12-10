#!/usr/bin/env python
"""
Clean up training run directories.

Two modes:
1. Default mode: Remove runs WITHOUT checkpoints (failed runs)
2. Interactive mode: Review runs WITH checkpoints and choose which to delete
"""

import argparse
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import yaml


class RunInfo:
    """Information about a training run."""

    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.name = run_dir.name
        self.checkpoints = self._find_checkpoints()
        self.has_checkpoints = len(self.checkpoints) > 0
        self.size_mb = self._get_dir_size()
        self.config = self._load_config()
        self.tags = self._extract_tags()
        self.overrides = self._extract_overrides()
        self.datetime = self._extract_datetime()

    def _find_checkpoints(self) -> List[Path]:
        """Find all checkpoint files in the run directory."""
        checkpoint_extensions = [".ckpt", ".pt", ".pth"]
        checkpoints = []
        for ext in checkpoint_extensions:
            checkpoints.extend(self.run_dir.rglob(f"*{ext}"))
        return sorted(checkpoints)

    def _get_dir_size(self) -> float:
        """Calculate total directory size in MB."""
        total_size = 0
        for item in self.run_dir.rglob("*"):
            if item.is_file():
                total_size += item.stat().st_size
        return total_size / (1024 * 1024)  # Convert to MB

    def _load_config(self) -> Dict:
        """Load Hydra config if available."""
        config_path = self.run_dir / ".hydra" / "config.yaml"
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    return yaml.safe_load(f)
            except Exception as e:
                print(f"Warning: Could not load config for {self.name}: {e}")
        return {}

    def _extract_tags(self) -> List[str]:
        """Extract tags from config."""
        if self.config and "tags" in self.config:
            tags = self.config["tags"]
            if isinstance(tags, list):
                return tags
            elif isinstance(tags, str):
                return [tags]
        return []

    def _extract_overrides(self) -> List[str]:
        """Extract command line overrides from Hydra config."""
        overrides_path = self.run_dir / ".hydra" / "overrides.yaml"
        if overrides_path.exists():
            try:
                with open(overrides_path, "r") as f:
                    overrides = yaml.safe_load(f)
                    if isinstance(overrides, list):
                        return overrides
            except Exception:
                pass
        return []

    def _extract_datetime(self) -> str:
        """Extract datetime from directory name or modification time."""
        # Try to parse from directory name (common format: YYYY-MM-DD_HH-MM-SS)
        name_parts = self.name.split("_")
        for part in name_parts:
            if len(part) == 10 and part.count("-") == 2:  # Date format
                return part

        # Fall back to directory creation time
        timestamp = self.run_dir.stat().st_mtime
        return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")

    def display_summary(self, verbose: bool = False):
        """Display information about this run."""
        print(f"\n{'='*80}")
        print(f"Run: {self.name}")
        print(f"{'='*80}")
        print(f"Date: {self.datetime}")
        print(f"Size: {self.size_mb:.2f} MB")
        print(f"Path: {self.run_dir}")

        if self.tags:
            print(f"Tags: {', '.join(self.tags)}")

        if self.overrides:
            print(f"Overrides: {' '.join(self.overrides)}")

        if self.has_checkpoints:
            print(f"\nCheckpoints ({len(self.checkpoints)}):")
            for ckpt in self.checkpoints:
                size_mb = ckpt.stat().st_size / (1024 * 1024)
                rel_path = ckpt.relative_to(self.run_dir)
                print(f"  - {rel_path} ({size_mb:.2f} MB)")
        else:
            print("\nNo checkpoints found")

        if verbose and self.config:
            print("\nConfig summary:")
            if "model" in self.config:
                print(f"  Model: {self.config.get('model', {}).get('_target_', 'unknown')}")
            if "data" in self.config:
                print(f"  Data: {self.config.get('data', {}).get('_target_', 'unknown')}")
            if "trainer" in self.config:
                trainer = self.config["trainer"]
                if isinstance(trainer, dict):
                    max_epochs = trainer.get("max_epochs", "unknown")
                    print(f"  Max epochs: {max_epochs}")


def find_all_runs(base_dir: Path) -> List[RunInfo]:
    """Find all run directories."""
    runs = []

    # Look for directories that contain .hydra folders or checkpoint files
    # This helps identify actual run directories vs intermediate folders
    for item in base_dir.rglob("*"):
        if item.is_dir():
            # Check if this looks like a run directory
            has_hydra = (item / ".hydra").exists()
            has_checkpoints = (
                any(item.rglob("*.ckpt")) or any(item.rglob("*.pt")) or any(item.rglob("*.pth"))
            )

            if has_hydra or has_checkpoints:
                # Make sure we haven't already added a parent directory
                is_subdirectory = any(item.is_relative_to(r.run_dir) for r in runs)
                if not is_subdirectory:
                    runs.append(RunInfo(item))

    return sorted(runs, key=lambda r: r.datetime)


def cleanup_failed_runs(base_dir: Path, delete: bool = False):
    """Remove runs without checkpoints."""
    print(f"\nScanning for runs in: {base_dir}")
    runs = find_all_runs(base_dir)

    failed_runs = [r for r in runs if not r.has_checkpoints]
    successful_runs = [r for r in runs if r.has_checkpoints]

    print(f"\nFound {len(runs)} total runs:")
    print(f"  - {len(successful_runs)} with checkpoints (will be kept)")
    print(f"  - {len(failed_runs)} without checkpoints (candidates for deletion)")

    if not failed_runs:
        print("\nNo failed runs to clean up!")
        return

    total_size_mb = sum(r.size_mb for r in failed_runs)
    print(f"\nTotal space to reclaim: {total_size_mb:.2f} MB ({total_size_mb/1024:.2f} GB)")

    print("\nRuns without checkpoints:")
    for run in failed_runs:
        print(f"  - {run.name} ({run.size_mb:.2f} MB) - {run.datetime}")

    if not delete:
        print("\n" + "=" * 80)
        print("DRY RUN MODE - No files will be deleted")
        print("To actually delete these runs, use --delete flag")
        print("=" * 80)
        return

    # Confirm deletion
    print("\n" + "=" * 80)
    print("WARNING: This will permanently delete the directories listed above!")
    print("=" * 80)
    response = input(f"\nDelete {len(failed_runs)} runs without checkpoints? (yes/no): ")

    if response.lower() != "yes":
        print("Deletion cancelled.")
        return

    # Double confirmation
    response2 = input(f"Are you absolutely sure? Type 'DELETE' to confirm: ")
    if response2 != "DELETE":
        print("Deletion cancelled.")
        return

    # Perform deletion
    deleted_count = 0
    deleted_size = 0
    for run in failed_runs:
        try:
            shutil.rmtree(run.run_dir)
            deleted_count += 1
            deleted_size += run.size_mb
            print(f"✓ Deleted: {run.name}")
        except Exception as e:
            print(f"✗ Failed to delete {run.name}: {e}")

    print(f"\n{'='*80}")
    print(
        f"Deleted {deleted_count} runs, reclaimed {deleted_size:.2f} MB ({deleted_size/1024:.2f} GB)"
    )
    print(f"{'='*80}")


def interactive_cleanup(base_dir: Path):
    """Interactively review and delete runs with checkpoints."""
    print(f"\nScanning for runs in: {base_dir}")
    runs = find_all_runs(base_dir)

    runs_with_checkpoints = [r for r in runs if r.has_checkpoints]

    print(f"\nFound {len(runs_with_checkpoints)} runs with checkpoints")

    if not runs_with_checkpoints:
        print("No runs to review!")
        return

    total_size = sum(r.size_mb for r in runs_with_checkpoints)
    print(f"Total size: {total_size:.2f} MB ({total_size/1024:.2f} GB)")

    deleted_count = 0
    deleted_size = 0

    for i, run in enumerate(runs_with_checkpoints, 1):
        run.display_summary(verbose=True)

        print(f"\n[{i}/{len(runs_with_checkpoints)}] What would you like to do?")
        print("  k = keep")
        print("  d = delete")
        print("  s = skip to next")
        print("  q = quit")

        while True:
            choice = input("\nChoice (k/d/s/q): ").lower().strip()

            if choice == "k":
                print(f"Keeping {run.name}")
                break

            elif choice == "d":
                # Confirm deletion
                confirm = input(
                    f"Delete {run.name} ({run.size_mb:.2f} MB)? Type 'yes' to confirm: "
                )
                if confirm.lower() == "yes":
                    try:
                        shutil.rmtree(run.run_dir)
                        deleted_count += 1
                        deleted_size += run.size_mb
                        print(f"✓ Deleted: {run.name}")
                    except Exception as e:
                        print(f"✗ Failed to delete: {e}")
                else:
                    print("Deletion cancelled, keeping run")
                break

            elif choice == "s":
                print("Skipping...")
                break

            elif choice == "q":
                print("\nExiting interactive mode...")
                if deleted_count > 0:
                    print(
                        f"\nDeleted {deleted_count} runs, reclaimed {deleted_size:.2f} MB ({deleted_size/1024:.2f} GB)"
                    )
                return

            else:
                print("Invalid choice. Please enter k, d, s, or q")

    print(f"\n{'='*80}")
    print(f"Interactive cleanup complete!")
    if deleted_count > 0:
        print(
            f"Deleted {deleted_count} runs, reclaimed {deleted_size:.2f} MB ({deleted_size/1024:.2f} GB)"
        )
    else:
        print("No runs were deleted.")
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(
        description="Clean up training run directories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run - preview what would be deleted (runs without checkpoints)
  python cleanup_failed_runs.py /path/to/runs

  # Actually delete runs without checkpoints
  python cleanup_failed_runs.py /path/to/runs --delete

  # Interactive mode - review runs with checkpoints
  python cleanup_failed_runs.py /path/to/runs --interactive
        """,
    )

    parser.add_argument(
        "--run_dir",
        type=Path,
        help="Path to the runs directory to clean up",
        default="./logs/train/runs",
        required=False,
    )

    parser.add_argument(
        "--delete",
        action="store_true",
        help="Actually delete files (default is dry-run)",
    )

    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive mode for runs with checkpoints",
    )

    args = parser.parse_args()

    if not args.run_dir.exists():
        print(f"Error: Directory not found: {args.run_dir}")
        return

    if not args.run_dir.is_dir():
        print(f"Error: Not a directory: {args.run_dir}")
        return

    if args.interactive:
        interactive_cleanup(args.run_dir)
    else:
        cleanup_failed_runs(args.run_dir, delete=args.delete)


if __name__ == "__main__":
    main()
