"""Mozilla Common Voice evaluation runner CLI for VoxPulse Voice Attribute Service.

Usage:
    uv run python scripts/evaluate.py \
        --dataset ./data/common_voice \
        --split test \
        --limit 100 \
        --api-url http://localhost:8000 \
        --output evaluation_results.json
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

from app.evaluation.dataset import CommonVoiceDataset, create_mock_common_voice_fixture
from app.evaluation.evaluator import Evaluator


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mozilla Common Voice Evaluation Harness for VoxPulse Voice Attribute Service"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="./data/common_voice",
        help="Path to Mozilla Common Voice dataset directory containing clips/ and TSV metadata (default: ./data/common_voice)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split name corresponding to <split>.tsv (e.g. test, validated, dev) (default: test)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (default: all)",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://localhost:8000",
        help="VoxPulse API base URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Filter samples by language/locale (e.g. 'en', 'es')",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save JSON evaluation report (e.g. evaluation_results.json)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="HTTP request timeout in seconds (default: 30.0)",
    )
    parser.add_argument(
        "--mock-if-missing",
        action="store_true",
        help="Auto-generate a small mock Common Voice test dataset if directory does not exist",
    )

    args = parser.parse_args()
    dataset_path = Path(args.dataset)

    # If dataset path doesn't exist and mock-if-missing or interactive check
    if not dataset_path.exists() or not (dataset_path / f"{args.split}.tsv").exists():
        if args.mock_if_missing:
            print(f"Dataset directory '{dataset_path}' not found. Generating mock test fixture...")
            create_mock_common_voice_fixture(dataset_path, n_samples=10)
        else:
            print(
                f"Error: Dataset path '{dataset_path}' or '{dataset_path / f'{args.split}.tsv'}' not found.\n"
                f"Please ensure you have placed your Mozilla Common Voice dataset in '{dataset_path}' "
                f"or pass '--mock-if-missing' to test with synthetic samples.",
                file=sys.stderr,
            )
            sys.exit(1)

    print(f"Initializing Common Voice dataset from: {dataset_path} (split: {args.split})")
    dataset = CommonVoiceDataset(
        dataset_dir=dataset_path,
        split=args.split,
        locale_filter=args.language,
        limit=args.limit,
    )

    evaluator = Evaluator(
        api_url=args.api_url,
        timeout=args.timeout,
    )

    print(f"Connecting to VoxPulse API at {args.api_url} ...")
    start_eval_time = time.perf_counter()

    def progress_callback(idx: int, total: int, result: Any) -> None:
        status_char = "[OK]" if result.error is None else "[FAIL]"
        if idx % 10 == 0 or idx == total or idx == 1:
            print(f"  [{idx}/{total}] Processed {result.filename} {status_char}")

    try:
        report = evaluator.run_evaluation(dataset, progress_callback=progress_callback)
    except Exception as e:
        print(f"Evaluation failed: {e}", file=sys.stderr)
        sys.exit(1)

    total_eval_elapsed = time.perf_counter() - start_eval_time
    report["summary"]["total_evaluation_time_seconds"] = round(total_eval_elapsed, 2)

    # Print Console Report
    console_output = evaluator.format_console_report(report)
    print("\n" + console_output)

    # Save JSON Report if requested
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\n[PASS] Saved JSON evaluation report to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
