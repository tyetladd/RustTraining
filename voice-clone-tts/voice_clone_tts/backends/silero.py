"""Silero TTS backend — preset voices, no cloning, native stress support.

This backend cannot copy the reference speaker (it only has built-in voices),
but it is the one that consumes silero-stress output *as is*: ``гот+ов`` is
exactly the format Silero TTS expects. It needs no license acknowledgement, is
small and fast on CPU, and makes a good baseline for comparing how much the
stress step actually changes Russian prosody.

Install::

    pip install 'voice-clone-tts[silero]'   # just torch
"""

from __future__ import annotations

import logging

import numpy as np

from voice_clone_tts.backends.base import SynthesisRequest, TTSBackend, register_backend
from voice_clone_tts.errors import BackendError, MissingDependencyError
from voice_clone_tts.text.stress import StressStyle

log = logging.getLogger(__name__)

# language code -> (torch.hub model id, default voice)
SILERO_MODELS = {
    "ru": ("v4_ru", "xenia"),
    "en": ("v3_en", "en_0"),
    "uk": ("v4_ua", "mykyta"),
}
SILERO_SAMPLE_RATES = (8_000, 24_000, 48_000)


@register_backend
class SileroBackend(TTSBackend):
    """Silero TTS (snakers4/silero-models) with preset speakers."""

    name = "silero"
    display_name = "Silero TTS (preset voices)"
    description = "Fast CPU TTS with native '+' stress support — does NOT clone the reference voice"
    supported_languages = set(SILERO_MODELS)
    requires_reference = False
    clones_voice = False
    native_stress_style = StressStyle.PLUS
    sample_rate = 48_000
    install_extra = "silero"

    def __init__(self, *, device: str = "auto", **options) -> None:
        super().__init__(device=device, **options)
        self.voice = options.get("voice")
        self.model_id = options.get("model_id")
        rate = int(options.get("sample_rate", 48_000))
        if rate not in SILERO_SAMPLE_RATES:
            raise BackendError(f"Silero supports {SILERO_SAMPLE_RATES} Hz, got {rate}")
        self.sample_rate = rate
        self._models: dict[str, object] = {}

    @classmethod
    def is_available(cls) -> bool:
        try:
            import torch  # noqa: F401
        except Exception:
            return False
        return True

    def _model_for(self, language_code: str):
        if language_code in self._models:
            return self._models[language_code]
        try:
            import torch
        except ImportError as exc:
            raise MissingDependencyError("torch", "silero", "Silero TTS") from exc
        if language_code not in SILERO_MODELS:
            raise BackendError(
                f"Silero TTS has no model for '{language_code}'. "
                f"Available: {', '.join(sorted(SILERO_MODELS))}"
            )
        model_id = self.model_id or SILERO_MODELS[language_code][0]
        log.info("loading silero_tts %s (%s)", model_id, language_code)
        model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-models",
            model="silero_tts",
            language=language_code,
            speaker=model_id,
            trust_repo=True,
        )
        model.to(torch.device(self.resolve_device()))
        self._models[language_code] = model
        return model

    def load(self) -> None:
        super().load()

    def close(self) -> None:
        self._models.clear()
        super().close()

    def synthesize(self, request: SynthesisRequest) -> np.ndarray:
        self.ensure_loaded()
        self.seed_everything(request.seed)
        language_code = request.language.code
        model = self._model_for(language_code)
        voice = self.voice or SILERO_MODELS[language_code][1]

        if request.profile is not None and request.profile.reference_files:
            log.debug("silero ignores the reference voice; speaking as '%s'", voice)

        # Text already carries "+" marks from silero-stress, so let the model
        # keep them instead of running its own (weaker) accentuation.
        audio = model.apply_tts(
            text=request.text,
            speaker=voice,
            sample_rate=self.sample_rate,
            put_accent=False,
            put_yo=False,
        )
        if hasattr(audio, "detach"):
            audio = audio.detach().cpu().numpy()
        wav = np.asarray(audio, dtype=np.float32).reshape(-1)
        if request.speed != 1.0:
            # Silero has no speed control: resample in time (pitch shifts too).
            from voice_clone_tts.audio import resample

            wav = resample(wav, int(self.sample_rate * request.speed), self.sample_rate)
        return wav
