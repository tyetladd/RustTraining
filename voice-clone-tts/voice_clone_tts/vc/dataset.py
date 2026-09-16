"""Building a training dataset for voice conversion.

Voice conversion trains on far more audio than XTTS conditioning needs: a
speaker profile keeps ~60 s of the cleanest speech, while RVC and so-vits-svc
want roughly 5-30 minutes cut into clips of up to ~10 s. So the dataset is
built from the *original* recording, not from the profile's reference clips,
and the profile is only used to find the source file and the speaker name.

The layout matches what both toolchains expect::

    <out_dir>/dataset_raw/<speaker>/clip_0001.wav   # so-vits-svc: dataset_raw root
                                                    # Applio/RVC: the speaker folder
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from voice_clone_tts import audio as audio_utils
from voice_clone_tts.errors import AudioError

log = logging.getLogger(__name__)

RAW_DIR = "dataset_raw"
RECOMMENDED_SECONDS = 300.0
"""Five minutes: below this, VC models copy the timbre only roughly."""


@dataclass
class TrainingDataset:
    """Clips prepared for a voice-conversion trainer."""

    directory: Path
    speaker: str
    clips: list[Path] = field(default_factory=list)
    sample_rate: int = 44_100
    total_duration: float = 0.0
    source: str | None = None
    median_f0: float = 0.0

    @property
    def raw_dir(self) -> Path:
        """Root that holds one folder per speaker (so-vits-svc style)."""
        return self.directory / RAW_DIR

    @property
    def speaker_dir(self) -> Path:
        """Folder with this speaker's clips (Applio/RVC style)."""
        return self.raw_dir / self.speaker

    def warnings(self) -> list[str]:
        issues: list[str] = []
        if self.total_duration < RECOMMENDED_SECONDS:
            issues.append(
                f"only {self.total_duration / 60:.1f} min of speech; "
                f"{RECOMMENDED_SECONDS / 60:.0f}-30 min gives a noticeably closer voice"
            )
        if len(self.clips) < 20:
            issues.append(f"only {len(self.clips)} clips; training may overfit quickly")
        return issues

    def describe(self) -> str:
        lines = [
            f"dataset     : {self.speaker_dir}",
            f"clips       : {len(self.clips)} ({self.total_duration / 60:.1f} min)",
            f"sample rate : {self.sample_rate} Hz",
        ]
        if self.median_f0:
            lines.append(f"median F0   : {self.median_f0:.0f} Hz")
        for warning in self.warnings():
            lines.append(f"warning     : {warning}")
        return "\n".join(lines)


def load_existing_dataset(
    out_dir: str | Path,
    speaker: str,
    *,
    sample_rate: int = 44_100,
    source: str | Path | None = None,
) -> TrainingDataset:
    """Describe clips that were prepared by an earlier run."""
    out_dir = Path(out_dir)
    speaker_dir = out_dir / RAW_DIR / speaker
    clips = sorted(speaker_dir.glob("*.wav"))
    if not clips:
        raise AudioError(f"no prepared clips in {speaker_dir}")

    total = 0.0
    for clip in clips:
        duration = audio_utils.probe_duration(clip)
        total += duration if duration else 0.0
    head = [audio_utils.load_audio(clip, sample_rate, mono=True)[0] for clip in clips[:20]]
    median_f0 = audio_utils.estimate_f0(np.concatenate(head), sample_rate) if head else 0.0

    log.info("reusing %d prepared clips (%.1f min) from %s", len(clips), total / 60, speaker_dir)
    return TrainingDataset(
        directory=out_dir,
        speaker=speaker,
        clips=clips,
        sample_rate=sample_rate,
        total_duration=total,
        source=str(source) if source else None,
        median_f0=median_f0,
    )


def _split_segment(start: float, end: float, max_clip_sec: float) -> list[tuple[float, float]]:
    """Cut one speech region into pieces of at most `max_clip_sec`."""
    duration = end - start
    if duration <= max_clip_sec:
        return [(start, end)]
    pieces = int(np.ceil(duration / max_clip_sec))
    step = duration / pieces
    return [(start + index * step, start + (index + 1) * step) for index in range(pieces)]


def build_training_dataset(
    source: str | Path | None = None,
    *,
    profile=None,
    out_dir: str | Path,
    speaker: str | None = None,
    sample_rate: int = 44_100,
    max_clip_sec: float = 10.0,
    min_clip_sec: float = 2.0,
    top_db: float = 35.0,
    target_dbfs: float = -20.0,
    overwrite: bool = False,
    reuse_existing: bool = False,
) -> TrainingDataset:
    """Cut a recording into training clips.

    Pass either `source` (the original recording) or `profile`; with a profile
    the original file is preferred and its reference clips are the fallback.

    With `reuse_existing` a dataset that is already on disk is picked up as is,
    which is what resuming an interrupted training run needs.
    """
    out_dir = Path(out_dir)
    if source is None and profile is None:
        raise AudioError("building a dataset needs either a source recording or a speaker profile")

    speaker = speaker or (profile.name if profile is not None else Path(str(source)).stem)
    if source is None:
        candidate = getattr(profile, "source_audio", None)
        if candidate and Path(candidate).exists():
            source = candidate
        else:
            log.warning(
                "original recording of profile '%s' is unavailable; falling back to its "
                "reference clips (%.0fs) — that is little for VC training",
                profile.name, getattr(profile, "total_duration", 0.0),
            )

    # A clip can never be shorter than half the maximum, otherwise a small
    # --max-clip-seconds would silently discard every clip.
    min_clip_sec = min(min_clip_sec, max_clip_sec / 2)

    speaker_dir = out_dir / RAW_DIR / speaker
    if speaker_dir.exists() and any(speaker_dir.iterdir()):
        if reuse_existing and not overwrite:
            return load_existing_dataset(out_dir, speaker, sample_rate=sample_rate, source=source)
        if not overwrite:
            raise AudioError(f"{speaker_dir} already holds clips; pass --overwrite to rebuild")
        shutil.rmtree(speaker_dir)
    speaker_dir.mkdir(parents=True, exist_ok=True)

    if source is not None:
        samples, sr = audio_utils.load_audio(source, sample_rate, mono=True)
        sources = [(samples, sr)]
    else:
        sources = [audio_utils.load_audio(path, sample_rate, mono=True)
                   for path in profile.reference_files]

    clips: list[Path] = []
    kept: list[np.ndarray] = []
    total = 0.0
    index = 0
    for samples, sr in sources:
        segments = audio_utils.detect_speech_segments(samples, sr, top_db=top_db)
        if not segments:
            segments = [audio_utils.SpeechSegment(0.0, samples.size / sr)]
        for segment in segments:
            for start, end in _split_segment(segment.start, segment.end, max_clip_sec):
                clip = audio_utils.slice_segment(samples, sr, start, end)
                clip = audio_utils.trim_silence(clip, sr, top_db=top_db)
                if clip.size < int(min_clip_sec * sr):
                    continue
                clip = audio_utils.normalize_loudness(clip, target_dbfs)
                clip = audio_utils.apply_fade(clip, sr, 8.0)
                index += 1
                path = speaker_dir / f"clip_{index:04d}.wav"
                audio_utils.save_audio(path, clip, sr)
                clips.append(path)
                kept.append(clip)
                total += clip.size / sr

    if not clips:
        raise AudioError(
            f"no usable speech found for the dataset "
            f"(source: {source or 'profile clips'}, clips shorter than "
            f"{min_clip_sec:.1f}s were dropped)"
        )

    median_f0 = audio_utils.estimate_f0(np.concatenate(kept[:50]), sample_rate)
    dataset = TrainingDataset(
        directory=out_dir,
        speaker=speaker,
        clips=clips,
        sample_rate=sample_rate,
        total_duration=total,
        source=str(source) if source else None,
        median_f0=median_f0,
    )
    for warning in dataset.warnings():
        log.warning("dataset: %s", warning)
    log.info("dataset ready: %d clips, %.1f min in %s", len(clips), total / 60, speaker_dir)
    return dataset
