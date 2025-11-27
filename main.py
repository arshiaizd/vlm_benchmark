from __future__ import annotations

import argparse
from pathlib import Path

from aftabe_vlm.runner import run_all_experiments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate VLMs on Aftabe-style picture word puzzles."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to dataset file (JSONL or CSV).",
    )
    parser.add_argument(
        "--results-db",
        type=str,
        default="results.sqlite3",
        help="Path to SQLite cache / results database.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional limit on number of samples (for debugging).",
    )
    parser.add_argument(
        "--retry-attempts",
        type=int,
        default=3,
        help="Max attempts for retry-with-feedback experiment (Experiment IV).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_all_experiments(
        dataset_path=Path(args.dataset),
        results_db_path=Path(args.results_db),
        max_samples=args.max_samples,
        retry_attempts=args.retry_attempts,
    )


if __name__ == "__main__":
    main()
