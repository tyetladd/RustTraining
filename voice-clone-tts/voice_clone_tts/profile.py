"""Speaker profiles: turn a reference recording into reusable voice data.

Building a profile is the "training" step of the pipeline. It is few-shot
speaker adaptation, not gradient fine-tuning: the reference MP3 is decoded,
transcribed, cut into the cleanest speech segments, loudness-normalized and
stored together with its metadata. Backends then derive their own conditioning
from it (XTTS caches GPT latents + a speaker embedding inside the profile
directory), which is why a profile is built once and reused for any number of
texts.

Layout on disk::

    profiles/anna/
        profile.json        metadata, quality report, chosen segments
        transcript.json     full ASR output (when ASR ran)
        refs/ref_01.wav     cleaned reference clips, 24 kHz mono
        cache/              backend-specific conditioning caches
"""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from voice_clone_tts import audio as audio_utils
from voice_clone_tts.asr import Transcript, WhisperTranscriber, select_reference_segments
from voice_clone_tts.audio import SpeechSegment
from voice_clone_tts.config import PipelineConfig
from voice_clone_tts.errors import AudioError, MissingDependencyError
from voice_clone_tts.text.languages import is_supported, normalize_code

log = logging.getLogger(__name__)

PROFILE_FILE = "profile.json"
TRANSCRIPT_FILE = "transcript.json"
REFS_DIR = "refs"
CACHE_DIR = "cache"
PROFILE_FORMAT = 1


@dataclass
class SpeakerProfile:
    """Everything the backends need to reproduce one voice."""

    name: str
    directory: Path
    language: str | None = None
    sample_rate: int = 24_000
    source_audio: str | None = None
    reference_files: list[Path] = field(default_factory=list)
    reference_text: str = ""
    transcript_text: str = ""
    total_duration: float = 0.0
    segments: list[SpeechSegment] = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    asr: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    created_at: str = ""
    is_temporary: bool = False

    # -- paths ------------------------------------------------------------
    @property
    def cache_dir(self) -> Path:
        path = self.directory / CACHE_DIR
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def primary_reference(self) -> Path:
        if not self.reference_files:
            raise AudioError(f"speaker profile '{self.name}' has no reference clips")
        return self.reference_files[0]

    def reference_paths(self) -> list[str]:
        return [str(path) for path in self.reference_files]

    # -- persistence ------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "format": PROFILE_FORMAT,
            "name": self.name,
            "language": self.language,
            "sample_rate": self.sample_rate,
            "source_audio": self.source_audio,
            "created_at": self.created_at,
            "total_duration": round(self.total_duration, 3),
            "reference_files": [path.name for path in self.reference_files],
            "reference_text": self.reference_text,
            "transcript_text": self.transcript_text,
            "segments": [segment.to_dict() for segment in self.segments],
            "stats": self.stats,
            "asr": self.asr,
            "warnings": self.warnings,
        }

    def save(self, directory: str | Path | None = None) -> Path:
        """Write the profile to disk and return its directory."""
        target = Path(directory) if directory else self.directory
        target.mkdir(parents=True, exist_ok=True)
        refs = target / REFS_DIR
        refs.mkdir(exist_ok=True)

        if target != self.directory:
            moved: list[Path] = []
            for source in self.reference_files:
                destination = refs / source.name
                if source.resolve() != destination.resolve():
                    shutil.copy2(source, destination)
                moved.append(destination)
            self.reference_files = moved
            self.directory = target

        (target / PROFILE_FILE).write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self.is_temporary = False
        return target

    @classmethod
    def load(cls, directory: str | Path) -> "SpeakerProfile":
        """Read a profile written by :meth:`save`."""
        directory = Path(directory)
        profile_path = directory / PROFILE_FILE
        if not profile_path.exists():
            raise AudioError(f"no {PROFILE_FILE} in {directory}")
        raw = json.loads(profile_path.read_text(encoding="utf-8"))
        if raw.get("format", 1) > PROFILE_FORMAT:
            raise AudioError(
                f"profile {directory} was written by a newer version "
                f"(format {raw['format']} > {PROFILE_FORMAT})"
            )
        refs = [directory / REFS_DIR / name for name in raw.get("reference_files", [])]
        missing = [path for path in refs if not path.exists()]
        if missing:
            raise AudioError(f"profile {directory} is missing reference clips: {[p.name for p in missing]}")
        return cls(
            name=raw.get("name", directory.name),
            directory=directory,
            language=raw.get("language"),
            sample_rate=int(raw.get("sample_rate", 24_000)),
            source_audio=raw.get("source_audio"),
            reference_files=refs,
            reference_text=raw.get("reference_text", ""),
            transcript_text=raw.get("transcript_text", ""),
            total_duration=float(raw.get("total_duration", 0.0)),
            segments=[SpeechSegment(**segment) for segment in raw.get("segments", [])],
            stats=raw.get("stats", {}),
            asr=raw.get("asr", {}),
            warnings=raw.get("warnings", []),
            created_at=raw.get("created_at", ""),
        )

    def cleanup(self) -> None:
        """Delete the directory if it was a throwaway temporary one."""
        if self.is_temporary and self.directory.exists():
            shutil.rmtree(self.directory, ignore_errors=True)

    def describe(self) -> str:
        lines = [
            f"speaker profile : {self.name}",
            f"directory       : {self.directory}",
            f"language        : {self.language or 'unknown'}",
            f"reference clips : {len(self.reference_files)} ({self.total_duration:.1f}s total)",
            f"sample rate     : {self.sample_rate} Hz",
        ]
        if self.asr:
            lines.append(f"asr             : {self.asr.get('model', '?')} "
                         f"(p={self.asr.get('language_probability', 0):.2f})")
        if self.transcript_text:
            preview = self.transcript_text[:120] + ("…" if len(self.transcript_text) > 120 else "")
            lines.append(f"transcript      : {preview}")
        for warning in self.warnings:
            lines.append(f"warning         : {warning}")
        return "\n".join(lines)


def _detect_language(transcript: Transcript | None, config: PipelineConfig) -> tuple[str, str]:
    """Return ``(language_code, how_it_was_decided)``."""
    if config.language:
        return normalize_code(config.language), "requested"
    if transcript and transcript.language:
        detected = transcript.language
        if is_supported(detected):
            return normalize_code(detected), "detected"
        log.warning(
            "Whisper detected '%s', which this pipeline does not support; falling back to '%s'",
            detected, config.fallback_language,
        )
    return normalize_code(config.fallback_language), "fallback"


def build_profile(
    source: str | Path,
    *,
    out_dir: str | Path | None = None,
    name: str | None = None,
    config: PipelineConfig | None = None,
    transcriber: WhisperTranscriber | None = None,
    overwrite: bool = False,
) -> SpeakerProfile:
    """Build a speaker profile from a reference recording (MP3, WAV, …).

    With `out_dir` omitted the profile lands in a temporary directory and
    :meth:`SpeakerProfile.cleanup` removes it.
    """
    config = config or PipelineConfig()
    reference_cfg = config.reference
    source = Path(source)
    name = name or source.stem

    if out_dir is None:
        directory = Path(tempfile.mkdtemp(prefix="vctts-profile-"))
        temporary = True
    else:
        directory = Path(out_dir)
        if directory.exists() and any(directory.iterdir()) and not overwrite:
            if (directory / PROFILE_FILE).exists():
                raise AudioError(
                    f"{directory} already holds a profile; pass --overwrite to rebuild it"
                )
        directory.mkdir(parents=True, exist_ok=True)
        temporary = False

    log.info("loading reference audio %s", source)
    samples, sample_rate = audio_utils.load_audio(source, reference_cfg.sample_rate, mono=True)
    stats = audio_utils.analyze(samples, sample_rate)
    if stats.peak_dbfs < -60.0:
        raise AudioError(
            f"{source.name} carries no audible signal (peak {stats.peak_dbfs:.1f} dBFS). "
            "Check that the file is not silent or muted."
        )
    warnings = stats.warnings()
    for warning in warnings:
        log.warning("reference audio: %s", warning)

    # ---- transcription / language detection -----------------------------
    transcript: Transcript | None = None
    if config.asr.enabled:
        transcriber = transcriber or WhisperTranscriber(config.asr)
        try:
            transcript = transcriber.transcribe(source, language=config.asr.language or config.language)
            log.info(
                "transcribed %d segments, language=%s (p=%.2f)",
                len(transcript.segments), transcript.language, transcript.language_probability,
            )
        except MissingDependencyError as exc:
            log.warning("%s\nContinuing with energy based segmentation.", exc)
        except Exception as exc:  # noqa: BLE001 - ASR must never break synthesis
            log.warning("transcription failed (%s); continuing with energy based segmentation", exc)

    language, decided_by = _detect_language(transcript, config)
    log.info("language: %s (%s)", language, decided_by)

    # ---- segment selection ----------------------------------------------
    if transcript and transcript.segments:
        candidates = [segment.to_speech_segment() for segment in transcript.segments]
    else:
        candidates = audio_utils.detect_speech_segments(
            samples, sample_rate, top_db=reference_cfg.silence_top_db
        )
        if not candidates:
            candidates = [SpeechSegment(0.0, samples.size / sample_rate)]

    chosen = select_reference_segments(
        candidates,
        min_sec=reference_cfg.min_segment_sec,
        max_sec=reference_cfg.max_segment_sec,
        target_total_sec=reference_cfg.target_total_sec,
        max_segments=reference_cfg.max_segments,
    )

    # ---- write reference clips ------------------------------------------
    refs_dir = directory / REFS_DIR
    if refs_dir.exists() and overwrite:
        shutil.rmtree(refs_dir)
    refs_dir.mkdir(parents=True, exist_ok=True)

    reference_files: list[Path] = []
    kept: list[SpeechSegment] = []
    total = 0.0
    for index, segment in enumerate(chosen, start=1):
        clip = audio_utils.slice_segment(samples, sample_rate, segment.start, segment.end)
        if reference_cfg.trim_silence:
            clip = audio_utils.trim_silence(clip, sample_rate, top_db=reference_cfg.silence_top_db)
        if clip.size < int(0.5 * sample_rate):
            continue
        clip = audio_utils.normalize_loudness(clip, reference_cfg.target_dbfs)
        clip = audio_utils.apply_fade(clip, sample_rate, 8.0)
        path = refs_dir / f"ref_{index:02d}.wav"
        audio_utils.save_audio(path, clip, sample_rate)
        reference_files.append(path)
        kept.append(segment)
        total += clip.size / sample_rate

    if not reference_files:
        # Last resort: use the whole (trimmed) recording as a single reference.
        clip = audio_utils.trim_silence(samples, sample_rate, top_db=reference_cfg.silence_top_db)
        if clip.size < int(0.5 * sample_rate):
            raise AudioError(
                f"{source.name} contains no usable speech "
                f"({stats.duration:.1f}s, RMS {stats.rms_dbfs:.1f} dBFS)"
            )
        clip = audio_utils.normalize_loudness(clip, reference_cfg.target_dbfs)
        path = refs_dir / "ref_01.wav"
        audio_utils.save_audio(path, clip, sample_rate)
        reference_files, kept = [path], [SpeechSegment(0.0, clip.size / sample_rate)]
        total = clip.size / sample_rate

    reference_text = " ".join(segment.text.strip() for segment in kept if segment.text.strip())

    profile = SpeakerProfile(
        name=name,
        directory=directory,
        language=language,
        sample_rate=sample_rate,
        source_audio=str(source.resolve()),
        reference_files=reference_files,
        reference_text=reference_text,
        transcript_text=transcript.text if transcript else "",
        total_duration=total,
        segments=kept,
        stats=stats.to_dict(),
        asr=(
            {
                "model": transcript.model,
                "language": transcript.language,
                "language_probability": transcript.language_probability,
                "segments": len(transcript.segments),
            }
            if transcript
            else {}
        ),
        warnings=warnings,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        is_temporary=temporary,
    )
    profile.save(directory)
    if transcript:
        (directory / TRANSCRIPT_FILE).write_text(
            json.dumps(transcript.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
    profile.is_temporary = temporary
    log.info("profile '%s': %d clips, %.1fs of speech", profile.name, len(reference_files), total)
    return profile


def load_reference_audio(profile: SpeakerProfile, sample_rate: int | None = None) -> np.ndarray:
    """Concatenate a profile's reference clips into one waveform."""
    clips: list[np.ndarray] = []
    for path in profile.reference_files:
        clip, _ = audio_utils.load_audio(path, sample_rate or profile.sample_rate, mono=True)
        clips.append(clip)
    if not clips:
        raise AudioError(f"profile '{profile.name}' has no reference clips")
    return np.concatenate(clips).astype(np.float32)
