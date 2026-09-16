"""Reference transcription and language detection (faster-whisper).

Two things come out of running ASR on the reference recording:

* the language, so the user does not have to pass ``--language``;
* a time-aligned transcript, which lets the pipeline pick the cleanest
  segments for voice conditioning and gives backends that need a reference
  *text* (F5-TTS style) something to work with.

The whole module is optional: with ``--no-asr`` the pipeline falls back to
energy based segmentation from :mod:`voice_clone_tts.audio`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from voice_clone_tts.audio import SpeechSegment
from voice_clone_tts.config import ASRConfig
from voice_clone_tts.errors import MissingDependencyError

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class TranscriptSegment:
    """One Whisper segment with the confidence signals we use for ranking."""

    start: float
    end: float
    text: str
    avg_logprob: float = 0.0
    no_speech_prob: float = 0.0
    compression_ratio: float = 1.0

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def quality(self) -> float:
        """Higher is better: confident speech, not silence, not a loop."""
        score = self.avg_logprob - 2.0 * self.no_speech_prob
        if self.compression_ratio > 2.4:  # Whisper's own repetition heuristic
            score -= 1.0
        return score

    def to_dict(self) -> dict:
        return {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "text": self.text,
            "avg_logprob": round(self.avg_logprob, 4),
            "no_speech_prob": round(self.no_speech_prob, 4),
        }

    def to_speech_segment(self) -> SpeechSegment:
        return SpeechSegment(start=self.start, end=self.end, text=self.text, score=self.quality)


@dataclass
class Transcript:
    """Result of transcribing the reference recording."""

    language: str
    language_probability: float
    segments: list[TranscriptSegment] = field(default_factory=list)
    model: str = ""
    duration: float = 0.0

    @property
    def text(self) -> str:
        return " ".join(segment.text.strip() for segment in self.segments if segment.text.strip())

    def to_dict(self) -> dict:
        return {
            "language": self.language,
            "language_probability": round(self.language_probability, 4),
            "model": self.model,
            "duration": round(self.duration, 3),
            "text": self.text,
            "segments": [segment.to_dict() for segment in self.segments],
        }


class WhisperTranscriber:
    """Lazy faster-whisper wrapper."""

    def __init__(self, config: ASRConfig | None = None) -> None:
        self.config = config or ASRConfig()
        self._model = None

    @staticmethod
    def is_available() -> bool:
        try:
            import faster_whisper  # noqa: F401
        except Exception:
            return False
        return True

    def _resolve_device(self) -> tuple[str, str]:
        device, compute_type = self.config.device, self.config.compute_type
        if device == "auto":
            device = "cpu"
            try:
                import torch

                if torch.cuda.is_available():
                    device = "cuda"
            except Exception:  # noqa: BLE001 - torch is optional here
                pass
        if compute_type == "default":
            compute_type = "float16" if device == "cuda" else "int8"
        return device, compute_type

    def load(self):
        if self._model is not None:
            return self._model
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise MissingDependencyError(
                "faster-whisper", "asr", "Reference transcription and language detection"
            ) from exc
        device, compute_type = self._resolve_device()
        log.info("loading Whisper '%s' on %s (%s)", self.config.model, device, compute_type)
        self._model = WhisperModel(self.config.model, device=device, compute_type=compute_type)
        return self._model

    def transcribe(self, audio_path: str | Path, *, language: str | None = None) -> Transcript:
        """Transcribe `audio_path`, detecting the language when not given."""
        model = self.load()
        language = language or self.config.language
        segments, info = model.transcribe(
            str(audio_path),
            language=language,
            beam_size=self.config.beam_size,
            vad_filter=self.config.vad_filter,
            word_timestamps=False,
        )
        collected = [
            TranscriptSegment(
                start=float(segment.start),
                end=float(segment.end),
                text=(segment.text or "").strip(),
                avg_logprob=float(getattr(segment, "avg_logprob", 0.0) or 0.0),
                no_speech_prob=float(getattr(segment, "no_speech_prob", 0.0) or 0.0),
                compression_ratio=float(getattr(segment, "compression_ratio", 1.0) or 1.0),
            )
            for segment in segments  # generator: this is where decoding happens
        ]
        return Transcript(
            language=getattr(info, "language", language or "") or "",
            language_probability=float(getattr(info, "language_probability", 0.0) or 0.0),
            segments=collected,
            model=self.config.model,
            duration=float(getattr(info, "duration", 0.0) or 0.0),
        )


def select_reference_segments(
    segments: list[SpeechSegment],
    *,
    min_sec: float = 3.0,
    max_sec: float = 15.0,
    target_total_sec: float = 60.0,
    max_segments: int = 8,
) -> list[SpeechSegment]:
    """Pick the best speech segments to condition the voice on.

    Short neighbours are merged until they reach `min_sec`, over-long segments
    are cut down to `max_sec`, and what is left is ranked by ASR confidence
    (energy-VAD segments all score 0, so they keep their original order) until
    `target_total_sec` of speech is collected.
    """
    merged: list[SpeechSegment] = []
    for segment in sorted(segments, key=lambda s: s.start):
        if segment.duration <= 0:
            continue
        if merged and merged[-1].duration < min_sec:
            previous = merged[-1]
            gap = segment.start - previous.end
            if gap < 1.0 and (segment.end - previous.start) <= max_sec:
                merged[-1] = SpeechSegment(
                    start=previous.start,
                    end=segment.end,
                    text=f"{previous.text} {segment.text}".strip(),
                    score=min(previous.score, segment.score),
                )
                continue
        merged.append(segment)

    trimmed = [
        SpeechSegment(s.start, min(s.end, s.start + max_sec), s.text, s.score)
        for s in merged
        if s.duration >= min(min_sec, 1.0)
    ]
    if not trimmed:
        # Nothing met the duration bar: keep the longest raw segment anyway.
        trimmed = sorted(segments, key=lambda s: s.duration, reverse=True)[:1]

    ranked = sorted(trimmed, key=lambda s: (-s.score, -s.duration))
    picked: list[SpeechSegment] = []
    total = 0.0
    for segment in ranked:
        if len(picked) >= max_segments or total >= target_total_sec:
            break
        picked.append(segment)
        total += segment.duration
    return sorted(picked, key=lambda s: s.start)
