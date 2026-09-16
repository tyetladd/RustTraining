"""Configuration objects for the synthesis pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from voice_clone_tts.text.stress import StressStyle

DEFAULT_SAMPLE_RATE = 24_000
"""Working sample rate; XTTS-v2 outputs 24 kHz."""


@dataclass
class ReferenceConfig:
    """How the reference recording is turned into a speaker profile."""

    sample_rate: int = DEFAULT_SAMPLE_RATE
    min_segment_sec: float = 3.0
    max_segment_sec: float = 15.0
    target_total_sec: float = 60.0
    """How much clean speech to keep; XTTS conditions well on 30-60 s."""
    max_segments: int = 8
    trim_silence: bool = True
    silence_top_db: float = 35.0
    target_dbfs: float = -23.0
    """Loudness normalization target for reference clips (approx. EBU R128)."""
    denoise: bool = False
    """Reserved: run a denoiser before profiling (needs an extra model)."""


@dataclass
class ASRConfig:
    """Whisper settings used to transcribe the reference recording."""

    enabled: bool = True
    model: str = "small"
    device: str = "auto"
    compute_type: str = "default"
    beam_size: int = 5
    vad_filter: bool = True
    language: str | None = None
    """Force a language instead of letting Whisper detect it."""


@dataclass
class TextConfig:
    """Text front-end settings."""

    normalize: bool = True
    stress: bool = True
    stress_style: StressStyle | str | None = None
    """``None`` means "use whatever the backend prefers"."""
    stress_device: str = "cpu"
    max_chars: int | None = None
    """``None`` means "use the language default"."""
    words_to_ignore: list[str] = field(default_factory=list)


@dataclass
class SynthesisConfig:
    """Backend and rendering settings."""

    backend: str = "xtts"
    speed: float = 1.0
    temperature: float = 0.7
    seed: int | None = None
    device: str = "auto"
    pause_between_chunks: float = 0.35
    """Seconds of silence inserted between chunks."""
    output_sample_rate: int | None = None
    """``None`` keeps the backend's native rate."""
    target_dbfs: float = -18.0
    fade_ms: float = 10.0
    backend_options: dict[str, Any] = field(default_factory=dict)

    # -- voice conversion (cloning on top of a backend with fixed voices) --
    voice_model: str | None = None
    """Directory of a trained voice-conversion model; ``None`` disables the stage."""
    converter: str | None = None
    """Converter name; ``None`` takes the one recorded in the voice model."""
    transpose: int | str = 0
    """Semitones to shift, or ``"auto"`` to match the target speaker's pitch."""
    converter_device: str = "auto"
    converter_options: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineConfig:
    """Everything the pipeline needs, in one object."""

    language: str | None = None
    """``None`` means "detect from the reference audio"."""
    reference: ReferenceConfig = field(default_factory=ReferenceConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    text: TextConfig = field(default_factory=TextConfig)
    synthesis: SynthesisConfig = field(default_factory=SynthesisConfig)
    fallback_language: str = "ru"

    def to_dict(self) -> dict:
        data = asdict(self)
        style = data["text"].get("stress_style")
        if isinstance(style, StressStyle):
            data["text"]["stress_style"] = style.value
        return data
