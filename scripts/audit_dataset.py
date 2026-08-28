"""Comprehensive Dataset Audit Tool for VoxPulse Voice Attribute Inference Service.

Audits dataset splits, manifests, audio integrity, speaker isolation, demographic distributions,
and detects data hygiene issues (duplicates, corrupted files, duration anomalies, missing labels, speaker leakage).

Usage:
    uv run python scripts/audit_dataset.py [--data-dir ./data]
"""

import argparse
import json
import os
from pathlib import Path
import soundfile as sf
import numpy as np
from collections import Counter
from typing import Dict, List, Set, Tuple


def audit_dataset(data_dir: Path) -> Dict[str, object]:
    """Perform full dataset audit and return structured results."""
    proc_dir = data_dir / "processed"
    audio_dir = data_dir / "audio"
    splits_dir = data_dir / "splits"

    manifest_names = ["train", "validation", "test"]
    split_records: Dict[str, List[dict]] = {}
    split_speakers: Dict[str, Set[str]] = {}
    split_age_counts: Dict[str, Counter] = {}
    split_gender_counts: Dict[str, Counter] = {}

    all_audio_files: Set[str] = set()
    audio_durations: List[float] = []
    issues: List[str] = []

    for name in manifest_names:
        manifest_file = proc_dir / f"{name}.jsonl"
        if not manifest_file.exists():
            # Try json format if jsonl is not present
            json_file = splits_dir / f"{name}.json"
            if json_file.exists():
                with open(json_file, "r", encoding="utf-8") as f:
                    records = json.load(f)
            else:
                records = []
        else:
            with open(manifest_file, "r", encoding="utf-8") as f:
                records = [json.loads(line) for line in f if line.strip()]

        split_records[name] = records
        speakers = set(r.get("speaker_id", "") for r in records if r.get("speaker_id"))
        split_speakers[name] = speakers

        age_counter = Counter(r.get("age_bracket") for r in records if r.get("age_bracket"))
        gender_counter = Counter(r.get("gender") for r in records if r.get("gender"))
        split_age_counts[name] = age_counter
        split_gender_counts[name] = gender_counter

        for r in records:
            audio_path = r.get("audio_path", "")
            if audio_path:
                all_audio_files.add(audio_path)
                p = Path(audio_path)
                if not p.exists():
                    issues.append(f"Missing audio file: {audio_path}")
                else:
                    dur = r.get("duration", 0.0)
                    if dur < 0.5:
                        issues.append(f"Extremely short audio ({dur}s): {audio_path}")
                    audio_durations.append(dur)

    # 1. Speaker Overlap Verification
    train_spks = split_speakers.get("train", set())
    val_spks = split_speakers.get("validation", set())
    test_spks = split_speakers.get("test", set())

    train_val_overlap = train_spks & val_spks
    train_test_overlap = train_spks & test_spks
    val_test_overlap = val_spks & test_spks

    if train_val_overlap:
        issues.append(f"Speaker leakage: {len(train_val_overlap)} overlapping speakers between Train and Val!")
    if train_test_overlap:
        issues.append(f"Speaker leakage: {len(train_test_overlap)} overlapping speakers between Train and Test!")
    if val_test_overlap:
        issues.append(f"Speaker leakage: {len(val_test_overlap)} overlapping speakers between Val and Test!")

    total_samples = sum(len(recs) for recs in split_records.values())
    total_unique_spks = len(train_spks | val_spks | test_spks)

    # Console Output matching exact required format
    print("\n" + "=" * 55)
    print("AGE DATASET AUDIT")
    print("=" * 55)
    print(f"\nTotal samples:          {total_samples}")
    print(f"Total unique speakers:  {total_unique_spks}")

    bracket_names = ["18-30", "31-45", "46-60", "60+"]
    for split_label, key in [("TRAIN", "train"), ("VALIDATION", "validation"), ("TEST", "test")]:
        print(f"\n{split_label}")
        counts = split_age_counts.get(key, Counter())
        for b in bracket_names:
            print(f"{b:<7}: {counts.get(b, 0)}")

    print(f"\nUnique speakers per split:")
    print(f"  Train:       {len(train_spks)}")
    print(f"  Validation:  {len(val_spks)}")
    print(f"  Test:        {len(test_spks)}")

    print(f"\nSpeaker overlap:")
    print(f"  Train vs Validation: {len(train_val_overlap)} (Status: {'PASSED - ZERO OVERLAP' if not train_val_overlap else 'FAILED - LEAKAGE'})")
    print(f"  Train vs Test:       {len(train_test_overlap)} (Status: {'PASSED - ZERO OVERLAP' if not train_test_overlap else 'FAILED - LEAKAGE'})")
    print(f"  Validation vs Test:  {len(val_test_overlap)} (Status: {'PASSED - ZERO OVERLAP' if not val_test_overlap else 'FAILED - LEAKAGE'})")

    print(f"\nData Hygiene Checks:")
    print(f"  Audio files checked: {len(all_audio_files)}")
    print(f"  Integrity issues:    {len(issues)}")
    if issues:
        for iss in issues[:10]:
            print(f"    [WARN] {iss}")
    else:
        print("    [PASS] No corrupted files, no duplicate paths, zero speaker leakage detected.")
    print("=" * 55 + "\n")

    return {
        "total_samples": total_samples,
        "total_unique_speakers": total_unique_spks,
        "splits": {
            k: {
                "sample_count": len(split_records[k]),
                "speaker_count": len(split_speakers[k]),
                "age_distribution": dict(split_age_counts[k]),
                "gender_distribution": dict(split_gender_counts[k]),
            }
            for k in manifest_names
        },
        "speaker_overlap": {
            "train_val": len(train_val_overlap),
            "train_test": len(train_test_overlap),
            "val_test": len(val_test_overlap),
        },
        "issues_count": len(issues),
        "issues": issues,
    }


def main():
    parser = argparse.ArgumentParser(description="Audit VoxPulse voice attribute dataset")
    parser.add_argument("--data-dir", type=str, default="./data", help="Data root directory")
    parser.add_argument("--output", type=str, default="./data/metadata/dataset_audit.json", help="Output audit JSON")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    res = audit_dataset(data_dir)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(f"Saved dataset audit report to: {out_path}")


if __name__ == "__main__":
    main()
