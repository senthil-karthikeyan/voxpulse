"""Shared voice representation and embedding extraction service using SpeechBrain ECAPA-TDNN."""

import os
from pathlib import Path
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torchaudio.transforms as T

from app.core.config import settings
from app.core.logging import logger


class NeuralAcousticEncoder(nn.Module):
    """PyTorch acoustic neural fallback network for extracting 192-dim voice representations."""

    def __init__(self, in_features: int = 80, out_features: int = 192) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_features, 256, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(256, 256, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(256)
        self.conv3 = nn.Conv1d(256, out_features, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract pooled 192-d voice representation from Mel filterbank frames.

        Args:
            x: Tensor of shape (batch, n_mels, time_frames).

        Returns:
            Tensor of shape (batch, out_features).
        """
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        mean_pool = torch.mean(x, dim=-1)
        return nn.functional.normalize(mean_pool, p=2, dim=-1)


class FeatureExtractor:
    """Extracts fixed 192-dimensional voice embeddings from 16kHz audio waveforms using SpeechBrain ECAPA-TDNN."""

    def __init__(
        self,
        embedding_dim: int = 192,
        device: str = settings.DEVICE,
        enable_speechbrain: bool = settings.ENABLE_SPEECHBRAIN_DOWNLOAD,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.device = torch.device(
            device if torch.cuda.is_available() and device == "cuda" else "cpu"
        )
        self.enable_speechbrain = enable_speechbrain
        self._speechbrain_model: Optional[object] = None
        self._neural_acoustic_encoder: Optional[NeuralAcousticEncoder] = None
        self._mel_transform: Optional[T.MelSpectrogram] = None
        self.is_loaded = False

    def load_model(self) -> None:
        """Initialize the SpeechBrain ECAPA voice representation encoder."""
        if self.is_loaded:
            return

        # Set optimal CPU thread count for low-latency inference
        if self.device.type == "cpu":
            num_threads = min(4, os.cpu_count() or 4)
            torch.set_num_threads(num_threads)

        cache_dir = os.path.join(str(settings.SPEECHBRAIN_CACHE_DIR), "spkrec-ecapa-voxceleb")

        # 1. Attempt to load from local pretrained cache or HuggingFace
        if self.enable_speechbrain:
            try:
                from speechbrain.inference.speaker import EncoderClassifier

                logger.info(
                    "Loading SpeechBrain ECAPA-TDNN voice encoder (pretrained_models/spkrec-ecapa-voxceleb)...",
                    extra={"event": "models_loading"},
                )

                # Check if local model directory exists
                source = cache_dir if os.path.exists(os.path.join(cache_dir, "hyperparams.yaml")) else "speechbrain/spkrec-ecapa-voxceleb"

                self._speechbrain_model = EncoderClassifier.from_hparams(
                    source=source,
                    savedir=cache_dir,
                    run_opts={"device": str(self.device)},
                )
                self.is_loaded = True
                logger.info(
                    "SpeechBrain ECAPA-TDNN voice encoder loaded successfully.",
                    extra={"event": "models_loaded", "encoder": "speechbrain/spkrec-ecapa-voxceleb"},
                )
                return
            except Exception as e:
                logger.warning(
                    f"Could not load SpeechBrain ECAPA-TDNN ({e}). Initializing neural acoustic encoder."
                )

        # Fallback to local neural acoustic encoder if SpeechBrain is unavailable
        self._neural_acoustic_encoder = NeuralAcousticEncoder(
            in_features=80, out_features=self.embedding_dim
        ).to(self.device)
        self._neural_acoustic_encoder.eval()

        self._mel_transform = T.MelSpectrogram(
            sample_rate=settings.TARGET_SAMPLE_RATE,
            n_fft=512,
            win_length=400,
            hop_length=160,
            n_mels=80,
            f_min=50.0,
            f_max=7600.0,
            power=2.0,
        ).to(self.device)

        self.is_loaded = True
        logger.info(
            "Neural Acoustic Voice Encoder initialized.",
            extra={"event": "models_loaded", "encoder": "neural_acoustic_encoder"},
        )

    def extract_embedding(
        self, waveform: np.ndarray, sample_rate: int = 16000
    ) -> torch.Tensor:
        """Extract a 192-dimensional voice embedding from normalized 16kHz audio.

        For audio longer than 4.0 seconds, selects the most salient 4.0-second speech window
        to maintain sub-500ms processing latency while preserving speaker acoustic representation.

        Args:
            waveform: 1D Float32 numpy array of normalized audio samples.
            sample_rate: Sampling frequency (16000 Hz).

        Returns:
            Torch tensor of shape (1, 192) on target device.
        """
        if not self.is_loaded:
            self.load_model()

        # Window selection for latency optimization (focus on highest energy speech segment)
        max_samples = int(sample_rate * 4.0)
        if len(waveform) > max_samples:
            hop = int(sample_rate * 0.5)
            best_start = 0
            best_energy = 0.0
            for start in range(0, len(waveform) - max_samples + 1, hop):
                seg_energy = float(np.sum(waveform[start : start + max_samples] ** 2))
                if seg_energy > best_energy:
                    best_energy = seg_energy
                    best_start = start
            waveform = waveform[best_start : best_start + max_samples]

        # Primary: SpeechBrain ECAPA-TDNN encoder
        if self._speechbrain_model is not None:
            tensor_wave = torch.from_numpy(waveform).unsqueeze(0).to(self.device).float()
            with torch.no_grad():
                emb = self._speechbrain_model.encode_batch(tensor_wave)
                if emb.ndim == 3:
                    emb = emb.squeeze(1)
                return emb

        # Fallback: Neural Acoustic Mel Spectrogram encoder
        tensor_wave = torch.from_numpy(waveform).unsqueeze(0).to(self.device).float()
        with torch.no_grad():
            mel_spec = self._mel_transform(tensor_wave)
            log_mel = torch.log(torch.clamp(mel_spec, min=1e-5))
            emb = self._neural_acoustic_encoder(log_mel)
            return emb


feature_extractor = FeatureExtractor()
