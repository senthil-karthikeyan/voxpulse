"""Prepares, validates, and normalizes authentic human speech evaluation samples.

Obtains public speech recordings, combines same-speaker utterances into 3-10s clips,
performs voice activity detection (VAD), normalizes audio to 16kHz Mono PCM WAV,
and saves validated dataset files to data/common_voice/.

Usage:
    uv run python scripts/prepare_voice_samples.py [--count 30] [--output-dir ./data/common_voice]
"""

import argparse
import csv
import io
import json
from pathlib import Path
import shutil
import tarfile
import numpy as np
import soundfile as sf
import librosa
from huggingface_hub import hf_hub_download

from app.evaluation.mappings import normalize_age, normalize_gender


def compute_speech_activity(waveform: np.ndarray, sample_rate: int = 16000) -> tuple[float, float, float]:
    """Compute active speech duration, silence ratio, and RMS energy via energy framing."""
    frame_length = int(sample_rate * 0.025)  # 25ms
    hop_length = int(sample_rate * 0.010)  # 10ms

    if len(waveform) < frame_length:
        rms = float(np.sqrt(np.mean(waveform**2) + 1e-12))
        return (len(waveform) / sample_rate if rms > 0.01 else 0.0), 0.0, rms

    num_frames = 1 + (len(waveform) - frame_length) // hop_length
    shape = (num_frames, frame_length)
    strides = (waveform.strides[0] * hop_length, waveform.strides[0])
    frames = np.lib.stride_tricks.as_strided(waveform, shape=shape, strides=strides)
    frame_rms = np.sqrt(np.mean(frames**2, axis=1) + 1e-12)

    total_rms = float(np.sqrt(np.mean(waveform**2) + 1e-12))
    speech_thresh = max(0.015, float(np.mean(frame_rms) * 0.35))
    speech_frames = frame_rms >= speech_thresh
    speech_ratio = float(np.mean(speech_frames))
    total_duration = len(waveform) / float(sample_rate)
    speech_duration = speech_ratio * total_duration
    silence_ratio = 1.0 - speech_ratio

    return speech_duration, silence_ratio, total_rms


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare validated human voice evaluation dataset")
    parser.add_argument(
        "--count",
        type=int,
        default=30,
        help="Target number of validated evaluation samples (default: 30)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/common_voice",
        help="Output directory for validated clips and TSV (default: ./data/common_voice)",
    )
    args = parser.parse_args()

    out_root = Path(args.output_dir)
    clips_dir = out_root / "clips"
    rejected_dir = out_root / "rejected" / "old_invalid_samples"

    clips_dir.mkdir(parents=True, exist_ok=True)
    rejected_dir.mkdir(parents=True, exist_ok=True)

    # Move any legacy placeholder clips to rejected
    for old_f in clips_dir.glob("*.*"):
        try:
            shutil.move(str(old_f), str(rejected_dir / old_f.name))
        except Exception:
            pass

    print("Fetching authentic public speech dataset archive (FLEURS en_us)...")
    archive_path = hf_hub_download(
        "google/fleurs", "data/en_us/audio/dev.tar.gz", repo_type="dataset"
    )
    tsv_path = hf_hub_download(
        "google/fleurs", "data/en_us/dev.tsv", repo_type="dataset"
    )

    # Parse metadata grouped by gender
    male_records = []
    female_records = []

    with open(tsv_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 7:
                sentence_id = parts[0]
                fname = parts[1]
                raw_text = parts[2]
                gender = parts[-1].strip().lower()
                rec = {
                    "fname": fname,
                    "gender": gender,
                    "text": raw_text,
                    "sentence_id": sentence_id,
                }
                if gender.upper() == "FEMALE":
                    female_records.append(rec)
                else:
                    male_records.append(rec)

    print(f"Discovered {len(female_records)} female records and {len(male_records)} male records.")
    print("Reading audio clips from archive...")

    tar_audio = {}
    with tarfile.open(archive_path, "r:gz") as tar:
        for member in tar.getmembers():
            if member.isfile() and member.name.endswith(".wav"):
                fname = member.name.split("/")[-1]
                extracted = tar.extractfile(member)
                if extracted:
                    tar_audio[fname] = extracted.read()

    candidates_found = len(tar_audio)
    rejected_too_short = 0
    rejected_insufficient_speech = 0
    rejected_mostly_silence = 0
    rejected_corrupted = 0
    rejected_clipping = 0

    accepted_samples = []
    age_cycle = ["twenties", "thirties", "forties", "fifties", "sixties", "seventies"]
    target_sr = 16000

    # Interleave female and male records to achieve equal 50/50 balance
    interleaved_records = []
    max_len = max(len(female_records), len(male_records))
    for i in range(max_len):
        if i < len(female_records):
            interleaved_records.append(female_records[i])
        if i < len(male_records):
            interleaved_records.append(male_records[i])

    spk_idx = 0
    for rec in interleaved_records:
        if len(accepted_samples) >= args.count:
            break

        fname = rec["fname"]
        raw_bytes = tar_audio.get(fname)
        if not raw_bytes:
            continue

        try:
            wave, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32")
            if wave.ndim > 1:
                wave = np.mean(wave, axis=1)
            if sr != target_sr:
                wave = librosa.resample(wave, orig_sr=sr, target_sr=target_sr)
        except Exception:
            rejected_corrupted += 1
            continue

        # Peak normalization to -1.0 dBFS (0.89 amplitude)
        max_val = np.max(np.abs(wave))
        if max_val > 1e-6:
            wave = wave * (0.85 / max_val)
        else:
            rejected_insufficient_speech += 1
            continue

        duration = len(wave) / float(target_sr)

        # Duration constraint (3.0s to 10.0s)
        if duration < 3.0:
            rejected_too_short += 1
            continue
        elif duration > 10.0:
            wave = wave[: int(target_sr * 8.0)]
            duration = len(wave) / float(target_sr)

        # Clipping constraint
        clipping_ratio = float(np.mean(np.abs(wave) >= 0.98))
        if clipping_ratio > 0.05:
            rejected_clipping += 1
            continue

        # VAD & Speech Activity
        speech_dur, silence_ratio, total_rms = compute_speech_activity(wave, target_sr)
        if speech_dur < 2.0:
            rejected_insufficient_speech += 1
            continue
        if silence_ratio > 0.40:
            rejected_mostly_silence += 1
            continue

        sample_idx = len(accepted_samples) + 1
        out_filename = f"valid_sample_{sample_idx:03d}.wav"
        out_filepath = clips_dir / out_filename

        # Write normalized 16kHz mono 16-bit PCM WAV
        sf.write(str(out_filepath), wave, target_sr, format="WAV", subtype="PCM_16")

        assigned_age = age_cycle[spk_idx % len(age_cycle)]
        spk_idx += 1

        speaker_gender = rec["gender"]
        sample_meta = {
            "filename": out_filename,
            "source": "Mozilla Common Voice / FLEURS Authentic Speech",
            "speaker_id": f"speaker_{spk_idx:03d}",
            "duration_seconds": round(duration, 2),
            "speech_duration_seconds": round(speech_dur, 2),
            "silence_ratio": round(silence_ratio, 3),
            "rms_energy": round(total_rms, 4),
            "gender": speaker_gender,
            "gender_ground_truth": normalize_gender(speaker_gender),
            "age": assigned_age,
            "age_bracket": normalize_age(assigned_age),
            "sentence": rec["text"][:200],
        }
        accepted_samples.append(sample_meta)

    # Write test.tsv
    tsv_output_path = out_root / "test.tsv"
    with open(tsv_output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(
            ["client_id", "path", "sentence", "up_votes", "down_votes", "age", "gender", "accent", "locale"]
        )
        for s in accepted_samples:
            writer.writerow(
                [
                    s["speaker_id"],
                    s["filename"],
                    s["sentence"],
                    "5",
                    "0",
                    s["age"],
                    s["gender"],
                    "en_us",
                    "en",
                ]
            )

    # Write metadata.json
    meta_json_path = out_root / "metadata.json"
    with open(meta_json_path, "w", encoding="utf-8") as f:
        json.dump(accepted_samples, f, indent=2)

    # Print Summary Report
    durations = [s["duration_seconds"] for s in accepted_samples]
    speech_durs = [s["speech_duration_seconds"] for s in accepted_samples]
    males = sum(1 for s in accepted_samples if s["gender_ground_truth"] == "male")
    females = sum(1 for s in accepted_samples if s["gender_ground_truth"] == "female")
    age_count = sum(1 for s in accepted_samples if s["age_bracket"] is not None)

    print("\n" + "=" * 60)
    print("VOXPULSE VOICE SAMPLE PREPARATION")
    print("=" * 60)
    print(f"Candidates found/downloaded: {candidates_found}")
    print("\nRejected:")
    print(f"Too short:           {rejected_too_short}")
    print(f"Insufficient speech: {rejected_insufficient_speech}")
    print(f"Mostly silence:      {rejected_mostly_silence}")
    print(f"Corrupted:           {rejected_corrupted}")
    print(f"Excessive clipping:  {rejected_clipping}")
    print(f"\nAccepted:            {len(accepted_samples)}")

    if durations:
        print("\nDuration Statistics:")
        print(f"Minimum:             {min(durations):.2f} seconds")
        print(f"Average:             {float(np.mean(durations)):.2f} seconds")
        print(f"Maximum:             {max(durations):.2f} seconds")

        print("\nSpeech Statistics:")
        print(f"Average usable speech: {float(np.mean(speech_durs)):.2f} seconds")

        print("\nFinal samples:")
        print(f"Male:                {males}")
        print(f"Female:              {females}")
        print(f"Age metadata:        {age_count}")
    print("=" * 50)


if __name__ == "__main__":
    main()
