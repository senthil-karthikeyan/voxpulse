"""Dataset loader and parser for Mozilla Common Voice TSV metadata."""

import csv
import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, List, Optional
import numpy as np
import soundfile as sf

from app.evaluation.mappings import normalize_age, normalize_gender


@dataclass
class DatasetSample:
    """Represents a validated Common Voice sample record."""

    audio_path: Path
    filename: str
    raw_gender: Optional[str] = None
    gender_ground_truth: Optional[str] = None
    raw_age: Optional[str] = None
    age_ground_truth: Optional[str] = None
    locale: Optional[str] = None
    sentence: Optional[str] = None


class CommonVoiceDataset:
    """Parser and iterator for local Mozilla Common Voice dataset directories."""

    def __init__(
        self,
        dataset_dir: Path | str,
        split: str = "test",
        locale_filter: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        self.locale_filter = locale_filter.strip().lower() if locale_filter else None
        self.limit = limit

    def resolve_tsv_path(self) -> Path:
        """Find the metadata TSV file for the specified split."""
        possible_names = [
            f"{self.split}.tsv",
            f"{self.split}.csv",
            f"{self.split}.txt",
        ]
        for name in possible_names:
            candidate = self.dataset_dir / name
            if candidate.exists() and candidate.is_file():
                return candidate

        raise FileNotFoundError(
            f"Could not find metadata file for split '{self.split}' in '{self.dataset_dir}'. "
            f"Expected one of: {', '.join(possible_names)}"
        )

    def resolve_audio_path(self, audio_filename: str) -> Optional[Path]:
        """Locate audio file in <dataset>/clips/ or directly in <dataset>/."""
        clean_name = audio_filename.strip()
        candidates = [
            self.dataset_dir / "clips" / clean_name,
            self.dataset_dir / clean_name,
        ]
        # Check without or with extension fallback
        for c in candidates:
            if c.exists() and c.is_file():
                return c
            # Try appending .mp3 or .wav if missing
            if not c.suffix:
                for ext in [".mp3", ".wav", ".ogg", ".flac"]:
                    c_ext = c.with_suffix(ext)
                    if c_ext.exists() and c_ext.is_file():
                        return c_ext
        return None

    def load_samples(self) -> List[DatasetSample]:
        """Parse the TSV file and return a list of verified DatasetSample objects."""
        tsv_path = self.resolve_tsv_path()
        samples: List[DatasetSample] = []
        skipped_missing_audio = 0
        skipped_malformed = 0

        with open(tsv_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f, delimiter="\t")
            try:
                header = next(reader)
            except StopIteration:
                return []

            # Clean header names
            norm_header = [h.strip().lower() for h in header]
            header_map = {col: idx for idx, col in enumerate(norm_header)}

            # Locate key columns dynamically
            path_idx = header_map.get("path") or header_map.get("filename") or header_map.get("audio") or 0
            gender_idx = header_map.get("gender")
            age_idx = header_map.get("age")
            locale_idx = header_map.get("locale") or header_map.get("accent")
            sentence_idx = header_map.get("sentence")

            for row_num, row in enumerate(reader, start=2):
                if not row or len(row) <= path_idx:
                    skipped_malformed += 1
                    continue

                raw_filename = row[path_idx].strip()
                if not raw_filename:
                    skipped_malformed += 1
                    continue

                raw_locale = row[locale_idx].strip() if locale_idx is not None and len(row) > locale_idx else None
                if self.locale_filter and raw_locale:
                    if self.locale_filter not in raw_locale.lower():
                        continue

                audio_path = self.resolve_audio_path(raw_filename)
                if not audio_path:
                    skipped_missing_audio += 1
                    continue

                raw_gender = row[gender_idx].strip() if gender_idx is not None and len(row) > gender_idx else None
                raw_age = row[age_idx].strip() if age_idx is not None and len(row) > age_idx else None
                sentence = row[sentence_idx].strip() if sentence_idx is not None and len(row) > sentence_idx else None

                sample = DatasetSample(
                    audio_path=audio_path,
                    filename=audio_path.name,
                    raw_gender=raw_gender,
                    gender_ground_truth=normalize_gender(raw_gender),
                    raw_age=raw_age,
                    age_ground_truth=normalize_age(raw_age),
                    locale=raw_locale,
                    sentence=sentence,
                )
                samples.append(sample)

                if self.limit and len(samples) >= self.limit:
                    break

        return samples


def create_mock_common_voice_fixture(
    target_dir: Path | str, n_samples: int = 10
) -> Path:
    """Generate a valid mock Common Voice directory with synthetic audio clips and TSVs for testing."""
    root = Path(target_dir)
    clips_dir = root / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    tsv_rows = [
        ["client_id", "path", "sentence", "up_votes", "down_votes", "age", "gender", "accent", "locale"]
    ]

    mock_labels = [
        ("male_20s.wav", "twenties", "male", 130.0),
        ("female_30s.wav", "thirties", "female", 220.0),
        ("male_50s.wav", "fifties", "male", 110.0),
        ("female_60s.wav", "sixties", "female", 195.0),
        ("teen_male.wav", "teens", "male", 160.0),  # Should exclude from age
        ("unknown_gender_40s.wav", "forties", "other", 180.0),  # Should exclude from gender
        ("female_70s.wav", "seventies", "female", 210.0),
        ("male_30s.wav", "thirties", "male", 125.0),
        ("female_20s.wav", "twenties", "female", 230.0),
        ("male_60s.wav", "sixties", "male", 105.0),
    ]

    sample_rate = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)

    for i in range(min(n_samples, len(mock_labels))):
        filename, age_lbl, gender_lbl, freq = mock_labels[i]
        audio_path = clips_dir / filename

        # Generate synthetic tone
        waveform = 0.6 * np.sin(2 * np.pi * freq * t) * (1 + 0.5 * np.sin(2 * np.pi * 3 * t))
        sf.write(str(audio_path), waveform.astype(np.float32), sample_rate, format="WAV", subtype="PCM_16")

        client_id = f"client_{i:04d}"
        tsv_rows.append(
            [client_id, filename, f"Sentence number {i}", "2", "0", age_lbl, gender_lbl, "us", "en"]
        )

    # Write test.tsv
    test_tsv = root / "test.tsv"
    with open(test_tsv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerows(tsv_rows)

    return root
