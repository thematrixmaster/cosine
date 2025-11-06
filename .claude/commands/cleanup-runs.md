---
description: Clean up training run directories (failed runs or interactive review)
---

Clean up training run directories in two modes:

## Mode 1: Clean up failed runs (no checkpoints)
Remove run directories that contain no checkpoint files (.ckpt, .pt, .pth).

**Important:** This will ONLY delete directories without any checkpoints. Directories with checkpoints are always kept safe.

Steps:
1. Run the cleanup script in dry-run mode to preview what would be deleted
2. Ask the user if they want to proceed with deletion
3. If confirmed, run with `--delete` flag to actually remove the directories

Usage:
- Dry run: `uv run python scripts/cleanup_failed_runs.py /scratch/users/stephen.lu/projects/protevo/logs/train/runs`
- Actual deletion: `uv run python scripts/cleanup_failed_runs.py /scratch/users/stephen.lu/projects/protevo/logs/train/runs --delete`

## Mode 2: Interactive review (runs with checkpoints)
Interactively review runs WITH checkpoints and decide which ones to delete.

For each run, the script displays:
- Run name and datetime
- Tags associated with the run
- Command/config overrides used to launch the run
- Checkpoint information (number of files, sizes)
- Config summary

The user can then choose to:
- Keep the run
- Delete the run (with confirmation)
- Skip to the next run
- Quit the interactive session

Usage:
- Interactive mode: `uv run python scripts/cleanup_failed_runs.py /scratch/users/stephen.lu/projects/protevo/logs/train/runs --interactive`

The script will:
- Scan all subdirectories in the specified path
- Check each for checkpoint files
- In regular mode: show summary and delete runs without checkpoints
- In interactive mode: let user review and decide on each run with checkpoints
- Perform double-checking before actual deletion to ensure safety
