#!/usr/bin/env python
"""Launch interpolation training using a trial entry from all_trials.json.

This picks a trial by index (default last) and forwards its hyperparameters
to `src_interpolation/trAISformer.py` via CLI arguments.
"""

import argparse
import json
import os
import shlex
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="Run trAISformer interpolation from a trial JSON entry")
    parser.add_argument("--trials-file", default="all_trials.json", help="Path to trials JSON file")
    parser.add_argument("--index", type=int, default=-1, help="Trial index (negative allowed, default last)")
    parser.add_argument("--max-epochs", type=int, default=None, help="Override max epochs for training")
    parser.add_argument("--python", default=sys.executable, help="Python executable to run (default: current) ")
    parser.add_argument("--dry-run", action="store_true", help="Only print the command that would be run")
    args = parser.parse_args()

    trials_path = os.path.abspath(args.trials_file)
    if not os.path.exists(trials_path):
        print(f"Trials file not found: {trials_path}")
        return 80

    with open(trials_path, "r", encoding="utf-8") as f:
        trials = json.load(f)

    if not trials:
        print("No trials found in file.")
        return 2

    idx = args.index
    if idx < 0:
        idx = len(trials) + idx

    if idx < 0 or idx >= len(trials):
        print(f"Index out of range: {args.index}")
        return 2

    trial = trials[idx]
    params = trial.get("params", {})

    lr = params.get("learning_rate")
    bs = params.get("batch_size")
    warmup = params.get("warmup_tokens")

    if lr is None or bs is None:
        print("Trial is missing required params (learning_rate, batch_size).")
        return 2

    max_epochs = args.max_epochs if args.max_epochs is not None else trial.get("epochs_run", 2)

    cmd = [
        args.python,
        "-u",
        os.path.join("src_interpolation", "trAISformer.py"),
        "--batch-size",
        str(bs),
        "--learning-rate",
        str(lr),
        "--warmup-tokens",
        str(warmup if warmup is not None else ""),
        "--max-epochs",
        str(max_epochs),
    ]

    # Remove any empty arguments (e.g., missing warmup)
    cmd = [c for c in cmd if c != ""]

    print("Running training with the following command:")
    print(" ".join(shlex.quote(c) for c in cmd))

    if args.dry_run:
        return 0

    try:
        return subprocess.call(cmd)
    except KeyboardInterrupt:
        print("Interrupted by user")
        return 130
    except Exception as e:
        print(f"Failed to run training: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
