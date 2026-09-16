"""Audio I/O and DSP helpers.

Only numpy, soundfile and soxr are required. MP3 decoding goes through
libsndfile when it was built with MPEG support (1.1+) and falls back to an
``ffmpeg`` subprocess otherwise, so neither pydub nor librosa is needed.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from voice_clone_tts.errors import AudioError

log = logging.getLogger(__name__)

EPS = 1e-12


# --------------------------------------------------------------------------
# Loading and saving
# --------------------------------------------------------------------------

def _ffmpeg_binary() -> str | None:
    return shutil.which("ffmpeg")


def has_ffmpeg() -> bool:
    return _ffmpeg_binary() is not None


def _load_with_soundfile(path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf

    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return data.T, sr  # (channels, samples)


def _load_with_ffmpeg(path: Path, sample_rate: int | None) -> tuple[np.ndarray, int]:
    binary = _ffmpeg_binary()
    if binary is None:
        raise AudioError(
            f"cannot decode {path.name}: libsndfile refused it and ffmpeg is not installed. "
            "Install ffmpeg (apt install ffmpeg / brew install ffmpeg) or convert the file to WAV."
        )
    sr = sample_rate or 24_000
    cmd = [
        binary, "-nostdin", "-v", "error", "-i", str(path),
        "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "1", "-ar", str(sr), "-",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise AudioError(f"ffmpeg failed to decode {path.name}: {proc.stderr.decode(errors='replace').strip()}")
    audio = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    return audio[np.newaxis, :], sr


def load_audio(
    path: str | Path,
    sample_rate: int | None = None,
    *,
    mono: bool = True,
) -> tuple[np.ndarray, int]:
    """Load any audio file as float32 in ``[-1, 1]``.

    Returns ``(samples, sample_rate)``; `samples` is 1-D when ``mono=True``.
    """
    path = Path(path)
    if not path.exists():
        raise AudioError(f"reference audio not found: {path}")
    if path.stat().st_size == 0:
        raise AudioError(f"reference audio is empty: {path}")

    try:
        data, sr = _load_with_soundfile(path)
    except Exception as exc:  # noqa: BLE001 - libsndfile raises its own types
        log.debug("soundfile could not read %s (%s), falling back to ffmpeg", path.name, exc)
        data, sr = _load_with_ffmpeg(path, sample_rate)

    if mono:
        data = to_mono(data)
    if sample_rate is not None and sr != sample_rate:
        data = resample(data, sr, sample_rate)
        sr = sample_rate
    if data.size == 0:
        raise AudioError(f"no audio samples decoded from {path}")
    return np.ascontiguousarray(data, dtype=np.float32), sr


def save_audio(path: str | Path, audio: np.ndarray, sample_rate: int) -> Path:
    """Write `audio` to `path`; the extension decides the format.

    MP3 is written by libsndfile when available, otherwise through ffmpeg.
    """
    import soundfile as sf

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.T  # soundfile expects (samples, channels)

    suffix = path.suffix.lower()
    if suffix in {".wav", ".flac", ".ogg", ".opus", ""}:
        sf.write(str(path), audio, sample_rate)
        return path

    try:
        sf.write(str(path), audio, sample_rate)
        return path
    except Exception as exc:  # noqa: BLE001
        log.debug("libsndfile cannot write %s (%s), trying ffmpeg", suffix, exc)

    binary = _ffmpeg_binary()
    if binary is None:
        raise AudioError(
            f"cannot write {suffix or 'this format'}: libsndfile refused it and ffmpeg is missing. "
            "Use a .wav output path or install ffmpeg."
        )
    tmp_wav = path.with_suffix(".tmp.wav")
    sf.write(str(tmp_wav), audio, sample_rate)
    try:
        cmd = [binary, "-nostdin", "-v", "error", "-y", "-i", str(tmp_wav), str(path)]
        proc = subprocess.run(cmd, capture_output=True)
        if proc.returncode != 0:
            raise AudioError(f"ffmpeg failed to encode {path.name}: {proc.stderr.decode(errors='replace').strip()}")
    finally:
        tmp_wav.unlink(missing_ok=True)
    return path


def probe_duration(path: str | Path) -> float | None:
    """Duration in seconds without decoding the whole file (best effort)."""
    path = Path(path)
    try:
        import soundfile as sf

        info = sf.info(str(path))
        if info.frames and info.samplerate:
            return info.frames / info.samplerate
    except Exception:  # noqa: BLE001
        pass
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        proc = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration",
             "-of", "json", str(path)],
            capture_output=True,
        )
        if proc.returncode == 0:
            try:
                return float(json.loads(proc.stdout)["format"]["duration"])
            except (KeyError, ValueError, json.JSONDecodeError):
                return None
    return None


# --------------------------------------------------------------------------
# Basic DSP
# --------------------------------------------------------------------------

def to_mono(audio: np.ndarray) -> np.ndarray:
    """Average a ``(channels, samples)`` array down to one channel."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 1:
        return audio
    return audio.mean(axis=0).astype(np.float32)


def resample(audio: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """High quality resampling via soxr (falls back to linear interpolation)."""
    if sr_in == sr_out:
        return np.asarray(audio, dtype=np.float32)
    audio = np.asarray(audio, dtype=np.float32)
    try:
        import soxr

        resampled = soxr.resample(audio.T if audio.ndim > 1 else audio, sr_in, sr_out, quality="HQ")
        return np.asarray(resampled.T if audio.ndim > 1 else resampled, dtype=np.float32)
    except ImportError:  # pragma: no cover - soxr is a hard dependency
        log.warning("soxr is not installed, using linear resampling")
        duration = audio.shape[-1] / sr_in
        target_len = int(round(duration * sr_out))
        src = np.linspace(0.0, duration, audio.shape[-1], endpoint=False)
        dst = np.linspace(0.0, duration, target_len, endpoint=False)
        if audio.ndim == 1:
            return np.interp(dst, src, audio).astype(np.float32)
        return np.stack([np.interp(dst, src, ch) for ch in audio]).astype(np.float32)


def dbfs(audio: np.ndarray) -> float:
    """RMS level in dBFS (``-inf`` guarded to -120)."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(audio)) + EPS))
    return float(20.0 * np.log10(max(rms, EPS)))


def peak_dbfs(audio: np.ndarray) -> float:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        return -120.0
    return float(20.0 * np.log10(max(float(np.max(np.abs(audio))), EPS)))


def normalize_loudness(audio: np.ndarray, target_dbfs: float = -23.0, *, peak_ceiling_db: float = -1.0) -> np.ndarray:
    """Scale to `target_dbfs` RMS, then clamp the peak below `peak_ceiling_db`."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        return audio
    gain = 10.0 ** ((target_dbfs - dbfs(audio)) / 20.0)
    out = audio * gain
    ceiling = 10.0 ** (peak_ceiling_db / 20.0)
    peak = float(np.max(np.abs(out)))
    if peak > ceiling:
        out = out * (ceiling / peak)
    return out.astype(np.float32)


def apply_fade(audio: np.ndarray, sample_rate: int, fade_ms: float = 10.0) -> np.ndarray:
    """Fade in/out to avoid clicks at chunk boundaries."""
    audio = np.asarray(audio, dtype=np.float32).copy()
    n = int(sample_rate * fade_ms / 1000.0)
    if n <= 0 or audio.size < 2 * n:
        return audio
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
    audio[:n] *= ramp
    audio[-n:] *= ramp[::-1]
    return audio


def silence(seconds: float, sample_rate: int) -> np.ndarray:
    return np.zeros(max(0, int(round(seconds * sample_rate))), dtype=np.float32)


def concat_with_pause(chunks: Sequence[np.ndarray], sample_rate: int, pause_sec: float = 0.35) -> np.ndarray:
    """Join waveforms with a fixed pause between them."""
    parts: list[np.ndarray] = []
    gap = silence(pause_sec, sample_rate)
    for index, chunk in enumerate(chunks):
        chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)
        if chunk.size == 0:
            continue
        if index and gap.size:
            parts.append(gap)
        parts.append(chunk)
    if not parts:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(parts).astype(np.float32)


# --------------------------------------------------------------------------
# Energy based silence handling
# --------------------------------------------------------------------------

def _frame_energy_db(audio: np.ndarray, frame: int, hop: int) -> np.ndarray:
    if audio.size < frame:
        return np.array([dbfs(audio)], dtype=np.float32)
    n_frames = 1 + (audio.size - frame) // hop
    strides = np.lib.stride_tricks.as_strided(
        audio, shape=(n_frames, frame), strides=(audio.strides[0] * hop, audio.strides[0])
    )
    rms = np.sqrt(np.mean(np.square(strides), axis=1) + EPS)
    return (20.0 * np.log10(np.maximum(rms, EPS))).astype(np.float32)


def trim_silence(
    audio: np.ndarray,
    sample_rate: int,
    *,
    top_db: float = 35.0,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
) -> np.ndarray:
    """Strip leading and trailing silence `top_db` below the loudest frame."""
    audio = np.ascontiguousarray(np.asarray(audio, dtype=np.float32).reshape(-1))
    if audio.size == 0:
        return audio
    frame = max(1, int(sample_rate * frame_ms / 1000.0))
    hop = max(1, int(sample_rate * hop_ms / 1000.0))
    energy = _frame_energy_db(audio, frame, hop)
    threshold = float(energy.max()) - top_db
    loud = np.flatnonzero(energy > threshold)
    if loud.size == 0:
        return audio
    start = int(loud[0]) * hop
    end = min(audio.size, int(loud[-1]) * hop + frame)
    return audio[start:end]


@dataclass(frozen=True)
class SpeechSegment:
    """A region of the reference recording that contains speech."""

    start: float
    end: float
    text: str = ""
    score: float = 0.0

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict:
        return {"start": self.start, "end": self.end, "text": self.text, "score": self.score}


def detect_speech_segments(
    audio: np.ndarray,
    sample_rate: int,
    *,
    top_db: float = 35.0,
    min_speech_sec: float = 0.4,
    min_silence_sec: float = 0.35,
    pad_sec: float = 0.1,
) -> list[SpeechSegment]:
    """Energy based VAD, used when Whisper is unavailable or disabled."""
    audio = np.ascontiguousarray(np.asarray(audio, dtype=np.float32).reshape(-1))
    if audio.size == 0:
        return []
    frame = max(1, int(sample_rate * 0.025))
    hop = max(1, int(sample_rate * 0.010))
    energy = _frame_energy_db(audio, frame, hop)
    threshold = float(energy.max()) - top_db
    voiced = energy > threshold

    segments: list[SpeechSegment] = []
    min_silence_frames = int(min_silence_sec * sample_rate / hop)
    start_frame: int | None = None
    silence_run = 0
    for index, is_voiced in enumerate(voiced):
        if is_voiced:
            if start_frame is None:
                start_frame = index
            silence_run = 0
            continue
        if start_frame is None:
            continue
        silence_run += 1
        if silence_run >= min_silence_frames:
            end_frame = index - silence_run
            segments.append((start_frame, end_frame))  # type: ignore[arg-type]
            start_frame, silence_run = None, 0
    if start_frame is not None:
        segments.append((start_frame, len(voiced) - 1))  # type: ignore[arg-type]

    total = audio.size / sample_rate
    out: list[SpeechSegment] = []
    for start_frame, end_frame in segments:  # type: ignore[misc]
        start = max(0.0, start_frame * hop / sample_rate - pad_sec)
        end = min(total, (end_frame * hop + frame) / sample_rate + pad_sec)
        if end - start >= min_speech_sec:
            out.append(SpeechSegment(start=start, end=end))
    return out


def slice_segment(audio: np.ndarray, sample_rate: int, start: float, end: float) -> np.ndarray:
    """Cut ``[start, end)`` seconds out of `audio`."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    i0 = max(0, int(round(start * sample_rate)))
    i1 = min(audio.size, int(round(end * sample_rate)))
    return audio[i0:i1] if i1 > i0 else np.zeros(0, dtype=np.float32)


# --------------------------------------------------------------------------
# Quality report
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class AudioStats:
    """Cheap quality metrics used to warn about a bad reference recording."""

    duration: float
    sample_rate: int
    rms_dbfs: float
    peak_dbfs: float
    clipping_ratio: float
    silence_ratio: float
    estimated_snr_db: float

    def to_dict(self) -> dict:
        return {
            "duration": round(self.duration, 3),
            "sample_rate": self.sample_rate,
            "rms_dbfs": round(self.rms_dbfs, 2),
            "peak_dbfs": round(self.peak_dbfs, 2),
            "clipping_ratio": round(self.clipping_ratio, 6),
            "silence_ratio": round(self.silence_ratio, 4),
            "estimated_snr_db": round(self.estimated_snr_db, 2),
        }

    def warnings(self, *, min_duration: float = 6.0) -> list[str]:
        """Human readable problems with the recording, worst first."""
        issues: list[str] = []
        if self.duration < min_duration:
            issues.append(
                f"reference is only {self.duration:.1f}s long; 30-60s of clean speech clones far better"
            )
        if self.clipping_ratio > 0.001:
            issues.append(f"{self.clipping_ratio * 100:.2f}% of samples are clipped; the clone will sound harsh")
        if self.estimated_snr_db < 15.0:
            issues.append(f"estimated SNR is {self.estimated_snr_db:.1f} dB; background noise will be cloned too")
        if self.silence_ratio > 0.5:
            issues.append(f"{self.silence_ratio * 100:.0f}% of the file is silence")
        if self.rms_dbfs < -40.0:
            issues.append(f"recording is very quiet ({self.rms_dbfs:.1f} dBFS RMS)")
        return issues


def analyze(audio: np.ndarray, sample_rate: int) -> AudioStats:
    """Compute :class:`AudioStats` for a waveform."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        return AudioStats(0.0, sample_rate, -120.0, -120.0, 0.0, 1.0, 0.0)

    frame = max(1, int(sample_rate * 0.025))
    hop = max(1, int(sample_rate * 0.010))
    energy = _frame_energy_db(audio, frame, hop)
    threshold = float(energy.max()) - 35.0
    voiced = energy > threshold
    # Noise floor from the quietest 10% of frames, signal from the voiced ones.
    quiet = np.sort(energy)[: max(1, energy.size // 10)]
    noise_db = float(np.mean(quiet))
    speech_db = float(np.mean(energy[voiced])) if voiced.any() else float(energy.max())

    return AudioStats(
        duration=audio.size / sample_rate,
        sample_rate=sample_rate,
        rms_dbfs=dbfs(audio),
        peak_dbfs=peak_dbfs(audio),
        clipping_ratio=float(np.mean(np.abs(audio) >= 0.999)),
        silence_ratio=float(1.0 - voiced.mean()) if voiced.size else 1.0,
        estimated_snr_db=max(0.0, speech_db - noise_db),
    )


def total_duration(segments: Iterable[SpeechSegment]) -> float:
    return float(sum(segment.duration for segment in segments))
