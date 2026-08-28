"""Inspects and reports complete dataset statistics for VoxPulse Voice Attribute Service.

Usage:
    uv run python scripts/inspect_age_dataset.py [--manifests-dir ./data/processed]
"""

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import numpy as np
import soundfile as sf


def analyze_split(manifest_path: Path) -> dict:
    if not manifest_path.exists():
        return {}

    records = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    total_samples = len(records)
    speakers = set(r.get("speaker_id") for r in records if r.get("speaker_id"))
    
    age_counts = Counter()
    gender_counts = Counter()
    spk_per_age = defaultdict(set)
    durations_per_age = defaultdict(list)
    gender_per_age = defaultdict(Counter)
    sources = Counter()

    for r in records:
        spk = r.get("speaker_id", "unknown")
        age = r.get("age_bracket", "unlabeled")
        gender = r.get("gender", "unlabeled")
        dur = r.get("duration", 0.0)
        src = r.get("source", "unknown")

        sources[src] += 1
        if age and age != "unlabeled":
            age_counts[age] += 1
            spk_per_age[age].add(spk)
            durations_per_age[age].append(dur)
            gender_per_age[age][gender] += 1
        
        if gender and gender != "unlabeled":
            gender_counts[gender] += 1

    stats = {
        "total_samples": total_samples,
        "unique_speakers": len(speakers),
        "speakers": list(speakers),
        "age_counts": dict(age_counts),
        "gender_counts": dict(gender_counts),
        "speakers_per_age": {k: len(v) for k, v in spk_per_age.items()},
        "durations_per_age": {
            k: {
                "count": len(v),
                "mean": round(float(np.mean(v)), 2) if v else 0,
                "median": round(float(np.median(v)), 2) if v else 0,
                "min": round(float(np.min(v)), 2) if v else 0,
                "max": round(float(np.max(v)), 2) if v else 0,
            }
            for k, v in durations_per_age.items()
        },
        "gender_per_age": {k: dict(v) for k, v in gender_per_age.items()},
        "sources": dict(sources),
    }
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Detailed Age & Gender Dataset Inspection Tool")
    parser.add_argument("--manifests-dir", type=str, default="./data/processed", help="Path to jsonl manifests")
    args = parser.parse_args()

    manifests_dir = Path(args.manifests_dir)
    splits = ["train", "validation", "test"]

    split_stats = {}
    for s in splits:
        p = manifests_dir / f"{s}.jsonl"
        if p.exists():
            split_stats[s] = analyze_split(p)

    print("=" * 80)
    print("           VOXPULSE VOICE ATTRIBUTE DATASET QUALITY AUDIT")
    print("=" * 80)

    total_samples = sum(s.get("total_samples", 0) for s in split_stats.values())
    all_speakers = set()
    for s in split_stats.values():
        all_speakers.update(s.get("speakers", []))

    print(f"Total Audio Samples:     {total_samples}")
    print(f"Total Unique Speakers:   {len(all_speakers)}")

    # Overlap Check
    train_spks = set(split_stats.get("train", {}).get("speakers", []))
    val_spks = set(split_stats.get("validation", {}).get("speakers", []))
    test_spks = set(split_stats.get("test", {}).get("speakers", []))

    overlap_tr_va = len(train_spks.intersection(val_spks))
    overlap_tr_te = len(train_spks.intersection(test_spks))
    overlap_va_te = len(val_spks.intersection(test_spks))

    print("\nSpeaker Overlap Audit:")
    print(f"  Train vs Val:   {overlap_tr_va} (Status: {'PASSED' if overlap_tr_va == 0 else 'LEAK DETECTED'})")
    print(f"  Train vs Test:  {overlap_tr_te} (Status: {'PASSED' if overlap_tr_te == 0 else 'LEAK DETECTED'})")
    print(f"  Val vs Test:    {overlap_va_te} (Status: {'PASSED' if overlap_va_te == 0 else 'LEAK DETECTED'})")

    age_brackets = ["18-30", "31-45", "46-60", "60+"]

    for split_name, stats in split_stats.items():
        print("\n" + "-" * 70)
        print(f"SPLIT: {split_name.upper()} ({stats['total_samples']} samples, {stats['unique_speakers']} unique speakers)")
        print("-" * 70)
        print(f"Sources: {stats.get('sources', {})}")
        print("\nAge Bracket Breakdown:")
        print(f"{'Bracket':<10} | {'Samples':<8} | {'Speakers':<9} | {'M / F / Unk':<18} | {'Mean Dur':<10} | {'Min - Max Dur':<15}")
        print("-" * 78)
        for b in age_brackets:
            cnt = stats["age_counts"].get(b, 0)
            spk_cnt = stats["speakers_per_age"].get(b, 0)
            g_dist = stats["gender_per_age"].get(b, {})
            m_cnt = g_dist.get("male", 0)
            f_cnt = g_dist.get("female", 0)
            u_cnt = g_dist.get("unlabeled", 0) + g_dist.get("unknown", 0)
            dur_info = stats["durations_per_age"].get(b, {})
            mean_dur = dur_info.get("mean", 0.0)
            min_dur = dur_info.get("min", 0.0)
            max_dur = dur_info.get("max", 0.0)
            print(f"{b:<10} | {cnt:<8} | {spk_cnt:<9} | {m_cnt:>3} / {f_cnt:>3} / {u_cnt:>3}     | {mean_dur:>5.2f}s    | {min_dur:.1f}s - {max_dur:.1f}s")

    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
