"""Audio quality assessment service for detecting duration, speech activity (VAD), clipping, and noise."""

from typing import Dict, List
import numpy as np

from app.core.config import settings
from app.schemas.response import AudioQualityEnum


class AudioQualityResult:
    """Detailed results of audio quality evaluation."""

    def __init__(
        self,
        quality: AudioQualityEnum,
        duration_seconds: float,
        rms_energy: float,
        speech_duration_seconds: float,
        speech_ratio: float,
        clipping_ratio: float,
        snr_db: float,
        reasons: List[str],
    ) -> None:
        self.quality = quality
        self.duration_seconds = round(duration_seconds, 2)
        self.rms_energy = round(rms_energy, 4)
        self.speech_duration_seconds = round(speech_duration_seconds, 2)
        self.speech_ratio = round(speech_ratio, 3)
        self.clipping_ratio = round(clipping_ratio, 4)
        self.snr_db = round(snr_db, 2)
        self.reasons = reasons

    def to_dict(self) -> Dict[str, object]:
        return {
            "quality": self.quality.value,
            "duration_seconds": self.duration_seconds,
            "rms_energy": self.rms_energy,
            "speech_duration_seconds": self.speech_duration_seconds,
            "speech_ratio": self.speech_ratio,
            "clipping_ratio": self.clipping_ratio,
            "snr_db": self.snr_db,
            "reasons": self.reasons,
        }


class AudioQualityService:
    """Assesses caller audio quality to determine if demographic inference can be performed reliably."""

    def __init__(
        self,
        min_duration_seconds: float = settings.MIN_AUDIO_DURATION_SECONDS,
        min_speech_duration_seconds: float = settings.MIN_SPEECH_DURATION_SECONDS,
        min_speech_ratio: float = settings.MIN_SPEECH_RATIO,
        min_rms_energy: float = settings.MIN_RMS_ENERGY,
        max_clipping_ratio: float = settings.MAX_CLIPPING_RATIO,
        min_snr_db: float = settings.MIN_SNR_DB,
        max_silence_ratio: float = settings.MAX_SILENCE_RATIO,
    ) -> None:
        self.min_duration_seconds = min_duration_seconds
        self.min_speech_duration_seconds = min_speech_duration_seconds
        self.min_speech_ratio = min_speech_ratio
        self.min_rms_energy = min_rms_energy
        self.max_clipping_ratio = max_clipping_ratio
        self.min_snr_db = min_snr_db
        self.max_silence_ratio = max_silence_ratio

    @staticmethod
    def _compute_frame_rms(waveform: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
        """Compute short-time RMS energy frames using fast vectorized numpy operations."""
        n_samples = len(waveform)
        if n_samples < frame_length:
            return np.array([float(np.sqrt(np.mean(waveform**2) + 1e-12))])

        num_frames = 1 + (n_samples - frame_length) // hop_length
        shape = (num_frames, frame_length)
        strides = (waveform.strides[0] * hop_length, waveform.strides[0])
        frames = np.lib.stride_tricks.as_strided(waveform, shape=shape, strides=strides)
        return np.sqrt(np.mean(frames**2, axis=1) + 1e-12)

    def evaluate(self, waveform: np.ndarray, sample_rate: int = 16000) -> AudioQualityResult:
        """Analyze waveform quality characteristics and compute Voice Activity Detection (VAD).

        Args:
            waveform: 1D Float32 numpy array representing normalized audio samples.
            sample_rate: Sampling rate (expected 16000 Hz).

        Returns:
            AudioQualityResult containing quality enum and diagnostic metrics.
        """
        reasons: List[str] = []
        is_insufficient = False
        is_degraded = False

        total_samples = len(waveform)
        if total_samples == 0:
            return AudioQualityResult(
                quality=AudioQualityEnum.INSUFFICIENT,
                duration_seconds=0.0,
                rms_energy=0.0,
                speech_duration_seconds=0.0,
                speech_ratio=0.0,
                clipping_ratio=0.0,
                snr_db=0.0,
                reasons=["Empty audio waveform"],
            )

        duration = total_samples / float(sample_rate)

        # 1. Check Total Audio Duration
        if duration < self.min_duration_seconds:
            is_insufficient = True
            reasons.append(
                f"Total audio duration ({duration:.2f}s) is below minimum required ({self.min_duration_seconds}s)"
            )

        # 2. Check Overall RMS Energy
        overall_rms = float(np.sqrt(np.mean(waveform**2) + 1e-12))
        if overall_rms < self.min_rms_energy:
            is_insufficient = True
            reasons.append(
                f"RMS energy ({overall_rms:.4f}) is below audible threshold ({self.min_rms_energy})"
            )

        # 3. Vectorized Voice Activity Detection (VAD) (25ms window, 10ms step)
        frame_length = int(sample_rate * 0.025)  # 25ms
        hop_length = int(sample_rate * 0.010)  # 10ms

        frame_rms = self._compute_frame_rms(waveform, frame_length, hop_length)

        # Dynamic speech threshold: energy higher than noise floor or minimum speech threshold
        speech_thresh = max(self.min_rms_energy, float(np.mean(frame_rms) * 0.35))
        speech_frames = frame_rms >= speech_thresh
        num_speech_frames = int(np.sum(speech_frames))
        total_frames = len(frame_rms)

        speech_ratio = num_speech_frames / float(max(total_frames, 1))
        speech_duration = speech_ratio * duration
        silence_ratio = 1.0 - speech_ratio

        # Check Active Speech Duration
        if speech_duration < self.min_speech_duration_seconds and not is_insufficient:
            is_insufficient = True
            reasons.append(
                f"Active speech duration ({speech_duration:.2f}s) is below minimum required ({self.min_speech_duration_seconds}s)"
            )
        elif speech_ratio < self.min_speech_ratio and not is_insufficient:
            is_insufficient = True
            reasons.append(
                f"Active speech ratio ({speech_ratio*100:.1f}%) is below minimum required ({self.min_speech_ratio*100:.1f}%)"
            )
        elif silence_ratio > self.max_silence_ratio:
            is_degraded = True
            reasons.append(
                f"High silence ratio ({silence_ratio*100:.1f}%) detected"
            )

        # 4. Clipping Detection
        clipping_threshold = 0.99
        clipped_samples = int(np.sum(np.abs(waveform) >= clipping_threshold))
        clipping_ratio = clipped_samples / float(total_samples)

        if clipping_ratio > self.max_clipping_ratio:
            is_degraded = True
            reasons.append(
                f"High clipping distortion ({clipping_ratio*100:.2f}% of samples saturated)"
            )

        # 5. SNR & Noise Floor Estimation
        sorted_frames = np.sort(frame_rms)
        noise_split = max(1, int(len(sorted_frames) * 0.25))
        speech_split = max(1, int(len(sorted_frames) * 0.75))

        noise_floor_rms = float(np.mean(sorted_frames[:noise_split]))
        signal_rms = float(np.mean(sorted_frames[speech_split:]))

        if noise_floor_rms > 1e-6:
            snr_db = 20.0 * np.log10(max(signal_rms, 1e-6) / noise_floor_rms)
        else:
            snr_db = 40.0

        if snr_db < self.min_snr_db and not is_insufficient:
            is_degraded = True
            reasons.append(
                f"Low Signal-to-Noise Ratio (estimated {snr_db:.1f} dB < {self.min_snr_db} dB)"
            )

        # Determine final audio quality
        if is_insufficient:
            quality = AudioQualityEnum.INSUFFICIENT
        elif is_degraded:
            quality = AudioQualityEnum.DEGRADED
        else:
            quality = AudioQualityEnum.GOOD

        return AudioQualityResult(
            quality=quality,
            duration_seconds=duration,
            rms_energy=overall_rms,
            speech_duration_seconds=speech_duration,
            speech_ratio=speech_ratio,
            clipping_ratio=clipping_ratio,
            snr_db=snr_db,
            reasons=reasons,
        )


audio_quality_service = AudioQualityService()
