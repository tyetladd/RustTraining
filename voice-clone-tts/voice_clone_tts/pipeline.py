"""The end-to-end pipeline: reference audio + text -> spoken audio.

::

    mp3  ──► decode ──► ASR (language + transcript) ──► segment pick ──► profile
                                                                          │
    text ──► normalize ──► stress (silero-stress) ──► chunk ──────────────┤
                                                                          ▼
                                                              backend.synthesize
                                                                          │
                                                        join · loudness · save
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from voice_clone_tts import audio as audio_utils
from voice_clone_tts.backends import DEFAULT_BACKEND, SynthesisRequest, TTSBackend, get_backend
from voice_clone_tts.config import PipelineConfig
from voice_clone_tts.errors import BackendError, TextError
from voice_clone_tts.profile import SpeakerProfile, build_profile
from voice_clone_tts.text.chunking import split_into_chunks
from voice_clone_tts.text.languages import LanguageSpec, get_language, normalize_code
from voice_clone_tts.text.normalize import normalize_text
from voice_clone_tts.text.stress import StressStyle, put_stress
from voice_clone_tts.vc import VoiceConverter, VoiceModel, get_converter

log = logging.getLogger(__name__)


@dataclass
class SynthesisResult:
    """Rendered audio plus everything that explains how it was produced."""

    audio: np.ndarray
    sample_rate: int
    language: str
    backend: str
    chunks: list[str]
    prepared_text: str
    profile: SpeakerProfile | None = None
    voice_model: str | None = None
    transpose: int = 0
    output_path: Path | None = None
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return float(self.audio.size) / self.sample_rate if self.sample_rate else 0.0

    def save(self, path: str | Path) -> Path:
        self.output_path = audio_utils.save_audio(path, self.audio, self.sample_rate)
        return self.output_path

    def describe(self) -> str:
        lines = [
            f"backend   : {self.backend}",
            f"language  : {self.language}",
            f"chunks    : {len(self.chunks)}",
            f"duration  : {self.duration:.2f}s @ {self.sample_rate} Hz",
        ]
        if self.voice_model:
            lines.append(f"voice     : {self.voice_model} (transpose {self.transpose:+d})")
        if self.output_path:
            lines.append(f"output    : {self.output_path}")
        if self.timings:
            stages = "  ".join(f"{k}={v:.2f}s" for k, v in self.timings.items())
            lines.append(f"timings   : {stages}")
        return "\n".join(lines)


def prepare_text(
    text: str,
    language: str | LanguageSpec,
    config: PipelineConfig | None = None,
    *,
    stress_style: StressStyle | str | None = None,
) -> tuple[str, list[str]]:
    """Normalize, stress and chunk `text`. Returns ``(prepared, chunks)``.

    `stress_style` is the caller's (usually the backend's) preference; a style
    set explicitly in ``config.text.stress_style`` overrides it.
    """
    config = config or PipelineConfig()
    spec = language if isinstance(language, LanguageSpec) else get_language(language)
    text_cfg = config.text

    if not text or not text.strip():
        raise TextError("nothing to say: the text is empty")

    prepared = normalize_text(text, spec, expand=text_cfg.normalize)
    if text_cfg.stress:
        # An explicit --stress-style wins; otherwise take the backend's preference.
        style = text_cfg.stress_style if text_cfg.stress_style is not None else stress_style
        style = StressStyle.parse(style) if style is not None else StressStyle.PLUS
        prepared = put_stress(
            prepared,
            spec,
            style=style,
            device=text_cfg.stress_device,
            **({"words_to_ignore": text_cfg.words_to_ignore} if text_cfg.words_to_ignore else {}),
        )
    max_chars = text_cfg.max_chars or spec.sentence_max_chars
    chunks = split_into_chunks(prepared, max_chars=max_chars, min_chars=max(20, max_chars // 8))
    if not chunks:
        raise TextError("nothing to say: the text contains no speakable characters")
    return prepared, chunks


def _resolve_profile(
    voice: str | Path | None,
    profile: SpeakerProfile | str | Path | None,
    config: PipelineConfig,
    backend: TTSBackend,
) -> SpeakerProfile | None:
    if isinstance(profile, SpeakerProfile):
        return profile
    if profile is not None:
        loaded = SpeakerProfile.load(profile)
        log.info("loaded speaker profile '%s' (%.1fs of reference speech)",
                 loaded.name, loaded.total_duration)
        return loaded
    if voice is not None:
        return build_profile(voice, config=config)
    if backend.requires_reference:
        raise BackendError(
            f"backend '{backend.name}' clones a voice, so it needs --voice <file.mp3> "
            "or --profile <dir>"
        )
    return None


def _resolve_language(
    requested: str | None,
    profile: SpeakerProfile | None,
    config: PipelineConfig,
) -> LanguageSpec:
    if requested:
        return get_language(requested)
    if config.language:
        return get_language(config.language)
    if profile and profile.language:
        return get_language(profile.language)
    return get_language(config.fallback_language)


def _resolve_converter(config: PipelineConfig) -> tuple[VoiceConverter, VoiceModel] | None:
    """Load the trained voice-conversion model, if one is configured."""
    synth_cfg = config.synthesis
    if not synth_cfg.voice_model:
        return None
    model = VoiceModel.load(synth_cfg.voice_model)
    name = synth_cfg.converter or model.converter
    if not name:
        raise BackendError(
            f"voice model {synth_cfg.voice_model} does not say which converter made it; "
            "pass --vc rvc|sovits"
        )
    converter = get_converter(
        name, device=synth_cfg.converter_device, **synth_cfg.converter_options
    )
    if not type(converter).is_available():
        raise BackendError(
            f"voice converter '{name}' is not usable here.\n{converter.install_hint}"
        )
    converter.load(model)
    return converter, model


def _resolve_transpose(setting: int | str, rendered: list[np.ndarray], sample_rate: int,
                       model: VoiceModel) -> int:
    """Turn ``transpose="auto"`` into semitones from synthesized to target pitch."""
    if setting != "auto":
        return int(setting)
    if not model.median_f0:
        log.warning("voice model '%s' has no reference pitch; using transpose 0", model.name)
        return 0
    source_f0 = audio_utils.estimate_f0(
        np.concatenate([chunk for chunk in rendered[:3] if chunk.size]), sample_rate
    )
    shift = audio_utils.semitones_between(source_f0, model.median_f0)
    shift = max(-24, min(24, shift))
    log.info("auto transpose: %.0f Hz -> %.0f Hz = %+d semitones",
             source_f0, model.median_f0, shift)
    return shift


def synthesize(
    voice: str | Path | None = None,
    text: str = "",
    *,
    language: str | None = None,
    profile: SpeakerProfile | str | Path | None = None,
    out_path: str | Path | None = None,
    config: PipelineConfig | None = None,
    backend: TTSBackend | str | None = None,
) -> SynthesisResult:
    """Speak `text` with the voice from `voice` (an MP3/WAV) or a saved profile.

    Everything else has a working default: the language is taken from the
    reference recording, the backend from the config, stress placement happens
    automatically for languages that have an accentor.
    """
    config = config or PipelineConfig()
    if language:
        config.language = normalize_code(language)
    timings: dict[str, float] = {}

    owns_backend = not isinstance(backend, TTSBackend)
    if isinstance(backend, TTSBackend):
        engine = backend
    else:
        name = backend or config.synthesis.backend or DEFAULT_BACKEND
        engine = get_backend(
            name, device=config.synthesis.device, **config.synthesis.backend_options
        )

    started = time.perf_counter()
    speaker = _resolve_profile(voice, profile, config, engine)
    timings["profile"] = time.perf_counter() - started

    spec = _resolve_language(language, speaker, config)
    if not engine.supports_language(spec):
        raise BackendError(
            f"backend '{engine.name}' does not support '{spec.code}' ({spec.name}). "
            f"Supported: {sorted(engine.supported_languages) if engine.supported_languages else 'any'}"
        )
    if speaker is not None and not engine.clones_voice:
        log.warning(
            "backend '%s' cannot clone voices — the reference recording only sets the language",
            engine.name,
        )

    started = time.perf_counter()
    prepared, chunks = prepare_text(text, spec, config, stress_style=engine.native_stress_style)
    timings["text"] = time.perf_counter() - started
    log.info("prepared %d chunk(s) in %s", len(chunks), spec.code)

    started = time.perf_counter()
    engine.ensure_loaded()
    conversion = _resolve_converter(config)
    timings["load"] = time.perf_counter() - started
    if conversion is None and speaker is not None and not engine.clones_voice:
        log.info("no --voice-model given, so the output keeps the backend's own voice")

    synth_cfg = config.synthesis
    rendered: list[np.ndarray] = []
    started = time.perf_counter()
    try:
        for index, chunk in enumerate(chunks, start=1):
            log.info("synthesizing chunk %d/%d (%d chars)", index, len(chunks), len(chunk))
            request = SynthesisRequest(
                text=chunk,
                language=spec,
                profile=speaker,
                speed=synth_cfg.speed,
                temperature=synth_cfg.temperature,
                seed=None if synth_cfg.seed is None else synth_cfg.seed + index,
                options=dict(synth_cfg.backend_options),
            )
            wav = np.asarray(engine.synthesize(request), dtype=np.float32).reshape(-1)
            if wav.size == 0:
                log.warning("chunk %d produced no audio, skipping", index)
                continue
            rendered.append(audio_utils.apply_fade(wav, engine.sample_rate, synth_cfg.fade_ms))
    finally:
        if owns_backend:
            engine.close()
    timings["synthesis"] = time.perf_counter() - started

    if not rendered:
        raise BackendError(f"backend '{engine.name}' returned no audio for this text")

    transpose = 0
    if conversion is not None:
        converter, voice_model = conversion
        started = time.perf_counter()
        transpose = _resolve_transpose(
            synth_cfg.transpose, rendered, engine.sample_rate, voice_model
        )
        try:
            rendered = [
                converter.convert(chunk, engine.sample_rate, voice_model, transpose=transpose)
                for chunk in rendered
            ]
        finally:
            converter.close()
        timings["conversion"] = time.perf_counter() - started
        log.info("converted %d chunk(s) into the voice of '%s'", len(rendered), voice_model.name)

    audio = audio_utils.concat_with_pause(rendered, engine.sample_rate, synth_cfg.pause_between_chunks)
    sample_rate = engine.sample_rate
    if synth_cfg.output_sample_rate and synth_cfg.output_sample_rate != sample_rate:
        audio = audio_utils.resample(audio, sample_rate, synth_cfg.output_sample_rate)
        sample_rate = synth_cfg.output_sample_rate
    if synth_cfg.target_dbfs is not None:
        audio = audio_utils.normalize_loudness(audio, synth_cfg.target_dbfs)

    result = SynthesisResult(
        audio=audio,
        sample_rate=sample_rate,
        language=spec.code,
        backend=engine.name,
        chunks=chunks,
        prepared_text=prepared,
        profile=speaker,
        voice_model=(conversion[1].name if conversion else None),
        transpose=transpose,
        timings=timings,
    )
    if out_path:
        result.save(out_path)
        log.info("wrote %s (%.2fs of audio)", result.output_path, result.duration)
    return result
