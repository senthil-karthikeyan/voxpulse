"""Audio decoding, normalization, and conversion pipeline."""

import io
import os
import shutil
import subprocess
import tempfile
from typing import Optional, Tuple
import numpy as np
import soundfile as sf
import librosa

from app.core.config import settings
from app.core.logging import logger


class AudioProcessingError(Exception):
    """Exception raised for audio decoding or normalization errors."""

    pass


class AudioProcessor:
    """Handles audio decoding, resampling to 16kHz mono, and normalization."""

    def __init__(self, target_sample_rate: int = settings.TARGET_SAMPLE_RATE) -> None:
        self.target_sample_rate = target_sample_rate
        self._ffmpeg_bin: Optional[str] = None

    def get_ffmpeg_binary(self) -> str:
        """Resolve the path to the ffmpeg executable."""
        if self._ffmpeg_bin:
            return self._ffmpeg_bin

        # 1. Check system PATH
        system_ffmpeg = shutil.which("ffmpeg")
        if system_ffmpeg:
            self._ffmpeg_bin = system_ffmpeg
            return self._ffmpeg_bin

        # 2. Fall back to imageio-ffmpeg bundled binary
        try:
            import imageio_ffmpeg

            self._ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
            return self._ffmpeg_bin
        except Exception as e:
            logger.warning(f"imageio_ffmpeg could not resolve ffmpeg binary: {e}")

        raise AudioProcessingError(
            "FFmpeg executable not found on system PATH or via imageio-ffmpeg."
        )

    def decode_and_normalize(self, audio_bytes: bytes) -> Tuple[np.ndarray, int]:
        """Decode raw audio bytes into 16kHz mono Float32 numpy array.

        Args:
            audio_bytes: Raw binary audio payload.

        Returns:
            Tuple of (waveform_1d_float32, sample_rate).

        Raises:
            AudioProcessingError: If audio is empty, corrupt, or unsupported.
        """
        if not audio_bytes or len(audio_bytes) == 0:
            raise AudioProcessingError("Audio payload is empty.")

        if len(audio_bytes) > settings.MAX_AUDIO_SIZE_BYTES:
            raise AudioProcessingError(
                f"Audio payload size ({len(audio_bytes)} bytes) exceeds limit ({settings.MAX_AUDIO_SIZE_BYTES} bytes)."
            )

        # 1. Attempt fast in-memory decoding with SoundFile / Librosa
        try:
            with io.BytesIO(audio_bytes) as bio:
                waveform, sr = sf.read(bio, dtype="float32", always_2d=False)
                # Convert multi-channel to mono
                if waveform.ndim > 1:
                    waveform = np.mean(waveform, axis=1)

                # Resample to target_sample_rate if needed
                if sr != self.target_sample_rate:
                    waveform = librosa.resample(
                        waveform, orig_sr=sr, target_sr=self.target_sample_rate
                    )

                waveform = self._sanitize_waveform(waveform)
                return waveform, self.target_sample_rate
        except Exception:
            # Fall back to FFmpeg for formats unsupported by libsndfile (e.g. mp3, m4a, aac, opus)
            pass

        # 2. Decode using FFmpeg
        return self._decode_via_ffmpeg(audio_bytes)

    def _decode_via_ffmpeg(self, audio_bytes: bytes) -> Tuple[np.ndarray, int]:
        """Decode audio bytes via FFmpeg subprocess to 16kHz mono float32 PCM."""
        ffmpeg_bin = self.get_ffmpeg_binary()

        # Try piping through stdin first
        try:
            cmd = [
                ffmpeg_bin,
                "-nostdin",
                "-threads",
                "1",
                "-i",
                "pipe:0",
                "-f",
                "f32le",
                "-acodec",
                "pcm_f32le",
                "-ar",
                str(self.target_sample_rate),
                "-ac",
                "1",
                "pipe:1",
            ]
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            stdout_data, stderr_data = process.communicate(
                input=audio_bytes, timeout=10.0
            )

            if process.returncode == 0 and stdout_data:
                waveform = np.frombuffer(stdout_data, dtype=np.float32)
                waveform = self._sanitize_waveform(waveform)
                return waveform, self.target_sample_rate
        except Exception:
            pass

        # Fall back to temporary file for container formats requiring seekable input (e.g. m4a/mp4)
        temp_dir = tempfile.mkdtemp(prefix="voxpulse_ffmpeg_")
        temp_in = os.path.join(temp_dir, "input_audio")
        temp_out = os.path.join(temp_dir, "output.raw")
        try:
            with open(temp_in, "wb") as f:
                f.write(audio_bytes)

            cmd = [
                ffmpeg_bin,
                "-y",
                "-nostdin",
                "-threads",
                "1",
                "-i",
                temp_in,
                "-f",
                "f32le",
                "-acodec",
                "pcm_f32le",
                "-ar",
                str(self.target_sample_rate),
                "-ac",
                "1",
                temp_out,
            ]
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=10.0,
                check=False,
            )

            if result.returncode != 0 or not os.path.exists(temp_out):
                err_msg = result.stderr.decode("utf-8", errors="replace")[:300]
                logger.warning(f"FFmpeg decoding failed: {err_msg}")
                raise AudioProcessingError("Unable to decode audio format or codec.")

            with open(temp_out, "rb") as f:
                pcm_data = f.read()

            if not pcm_data:
                raise AudioProcessingError("Decoded audio stream is empty.")

            waveform = np.frombuffer(pcm_data, dtype=np.float32)
            waveform = self._sanitize_waveform(waveform)
            return waveform, self.target_sample_rate
        except subprocess.TimeoutExpired:
            raise AudioProcessingError("Audio decoding timed out.")
        except AudioProcessingError:
            raise
        except Exception as e:
            raise AudioProcessingError(f"Audio processing error: {e}") from e
        finally:
            # Ephemeral guarantee: remove temporary files and directory immediately
            shutil.rmtree(temp_dir, ignore_errors=True)

    @staticmethod
    def _sanitize_waveform(waveform: np.ndarray) -> np.ndarray:
        """Ensure waveform is 1D float32, non-empty, and free of NaN/Inf."""
        if waveform.size == 0:
            raise AudioProcessingError("Decoded waveform contains 0 samples.")

        waveform = np.nan_to_num(waveform, nan=0.0, posinf=1.0, neginf=-1.0)
        waveform = waveform.astype(np.float32)

        # Clip extreme outliers to [-1.0, 1.0]
        waveform = np.clip(waveform, -1.0, 1.0)
        return waveform


audio_processor = AudioProcessor()
