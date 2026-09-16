"""XTTS-v2 backend — zero-shot voice cloning for Russian and English.

XTTS-v2 conditions on a few seconds of reference speech, so "training" is
really *few-shot speaker adaptation*: the GPT conditioning latents and the
speaker embedding are computed once from the profile's reference clips and
cached in ``<profile>/cache/xtts_latents.pt``. Every later synthesis reuses
them, which is both faster and more consistent than passing ``speaker_wav``
on each call.

Install::

    pip install 'voice-clone-tts[xtts]'      # coqui-tts fork + torch

License note: the XTTS-v2 *weights* are released under the Coqui Public Model
License (non-commercial). The pipeline refuses to download them until you
acknowledge that with ``VCTTS_ACCEPT_COQUI_LICENSE=1`` (or
``--accept-coqui-license``), which also sets Coqui's own ``COQUI_TOS_AGREED``.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np

from voice_clone_tts.backends.base import SynthesisRequest, TTSBackend, register_backend
from voice_clone_tts.errors import BackendError, MissingDependencyError
from voice_clone_tts.text.stress import StressStyle, strip_stress_marks

log = logging.getLogger(__name__)

DEFAULT_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
LICENSE_ENV = "VCTTS_ACCEPT_COQUI_LICENSE"

# Languages XTTS-v2 was trained on.
XTTS_LANGUAGES = {
    "en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar",
    "zh-cn", "ja", "hu", "ko", "hi",
}


@register_backend
class XTTSBackend(TTSBackend):
    """Coqui XTTS-v2 zero-shot cloning backend."""

    name = "xtts"
    display_name = "XTTS-v2 (Coqui)"
    description = "Zero-shot multilingual voice cloning, 24 kHz, ru/en and 15 more languages"
    supported_languages = XTTS_LANGUAGES
    requires_reference = True
    clones_voice = True
    # XTTS never saw "+" stress markers in training; ё survives and helps a lot.
    native_stress_style = StressStyle.NONE
    sample_rate = 24_000
    install_extra = "xtts"

    def __init__(self, *, device: str = "auto", **options) -> None:
        super().__init__(device=device, **options)
        self.model_name = options.get("model", DEFAULT_MODEL)
        self.gpt_cond_len = int(options.get("gpt_cond_len", 30))
        self.max_ref_length = int(options.get("max_ref_length", 60))
        self.repetition_penalty = float(options.get("repetition_penalty", 5.0))
        self.length_penalty = float(options.get("length_penalty", 1.0))
        self.top_k = int(options.get("top_k", 50))
        self.top_p = float(options.get("top_p", 0.85))
        self.enable_text_splitting = bool(options.get("enable_text_splitting", False))
        self._api = None
        self._model = None
        self._latents: dict[str, tuple] = {}

    @classmethod
    def is_available(cls) -> bool:
        try:
            import TTS  # noqa: F401
        except Exception:
            return False
        return True

    def supports_language(self, language) -> bool:
        code = language.xtts_code or language.code
        return code in XTTS_LANGUAGES

    # -- loading ----------------------------------------------------------
    @staticmethod
    def _check_license() -> None:
        if os.environ.get(LICENSE_ENV, "").lower() in {"1", "true", "yes"}:
            # Coqui's downloader asks interactively unless this is set.
            os.environ.setdefault("COQUI_TOS_AGREED", "1")
            return
        if os.environ.get("COQUI_TOS_AGREED", "").lower() in {"1", "true", "yes"}:
            return
        raise BackendError(
            "XTTS-v2 weights are distributed under the Coqui Public Model License "
            "(non-commercial use only: https://coqui.ai/cpml).\n"
            f"Acknowledge it with  export {LICENSE_ENV}=1  (or pass --accept-coqui-license) "
            "before the first run, or pick another backend with --backend."
        )

    def load(self) -> None:
        if self._model is not None:
            return
        self._check_license()
        try:
            from TTS.api import TTS as CoquiTTS
        except ImportError as exc:
            raise MissingDependencyError("coqui-tts", "xtts", "XTTS-v2 synthesis") from exc

        device = self.resolve_device()
        log.info("loading %s on %s (first run downloads ~1.8 GB)", self.model_name, device)
        self._api = CoquiTTS(model_name=self.model_name)
        try:
            self._api.to(device)
        except Exception as exc:  # noqa: BLE001 - older TTS releases use gpu=
            log.warning("could not move XTTS to %s (%s); staying on CPU", device, exc)
        self._model = getattr(getattr(self._api, "synthesizer", None), "tts_model", None)
        if self._model is not None:
            self.sample_rate = int(
                getattr(getattr(self._model, "config", None), "output_sample_rate", self.sample_rate)
                or self.sample_rate
            )
        super().load()

    def close(self) -> None:
        self._api = None
        self._model = None
        self._latents.clear()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
        super().close()

    # -- speaker conditioning ("training") --------------------------------
    def _conditioning(self, profile) -> tuple:
        """Compute or load the cached GPT latents + speaker embedding."""
        import torch

        key = str(profile.directory)
        if key in self._latents:
            return self._latents[key]

        cache_path: Path = profile.cache_dir / "xtts_latents.pt"
        references = profile.reference_paths()
        if cache_path.exists():
            try:
                blob = torch.load(cache_path, map_location="cpu", weights_only=False)
                if blob.get("references") == references:
                    latents = (blob["gpt_cond_latent"], blob["speaker_embedding"])
                    self._latents[key] = latents
                    log.info("reusing cached speaker latents from %s", cache_path)
                    return latents
                log.info("reference clips changed, recomputing speaker latents")
            except Exception as exc:  # noqa: BLE001 - a stale cache must not be fatal
                log.warning("ignoring unreadable latent cache %s (%s)", cache_path, exc)

        log.info("computing speaker latents from %d reference clip(s)", len(references))
        gpt_cond_latent, speaker_embedding = self._model.get_conditioning_latents(
            audio_path=references,
            gpt_cond_len=self.gpt_cond_len,
            max_ref_length=self.max_ref_length,
        )
        try:
            torch.save(
                {
                    "references": references,
                    "gpt_cond_latent": gpt_cond_latent.cpu(),
                    "speaker_embedding": speaker_embedding.cpu(),
                    "model": self.model_name,
                },
                cache_path,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("could not cache speaker latents (%s)", exc)
        self._latents[key] = (gpt_cond_latent, speaker_embedding)
        return gpt_cond_latent, speaker_embedding

    # -- synthesis --------------------------------------------------------
    def synthesize(self, request: SynthesisRequest) -> np.ndarray:
        self.ensure_loaded()
        self.seed_everything(request.seed)

        if request.profile is None:
            raise BackendError("the XTTS backend needs a reference voice; pass --voice")
        language_code = request.language.xtts_code or request.language.code
        if language_code not in XTTS_LANGUAGES:
            raise BackendError(
                f"XTTS-v2 does not speak '{request.language.code}'. "
                f"Supported: {', '.join(sorted(XTTS_LANGUAGES))}"
            )
        # "+" markers would be read as punctuation; ё stays.
        text = strip_stress_marks(request.text)

        if self._model is None:  # pragma: no cover - very old TTS releases
            wav = self._api.tts(
                text=text,
                speaker_wav=request.profile.reference_paths(),
                language=language_code,
                speed=request.speed,
            )
            return np.asarray(wav, dtype=np.float32).reshape(-1)

        gpt_cond_latent, speaker_embedding = self._conditioning(request.profile)
        device = self.resolve_device()
        if hasattr(gpt_cond_latent, "to"):
            gpt_cond_latent = gpt_cond_latent.to(device)
            speaker_embedding = speaker_embedding.to(device)

        result = self._model.inference(
            text=text,
            language=language_code,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            temperature=request.temperature,
            speed=request.speed,
            repetition_penalty=self.repetition_penalty,
            length_penalty=self.length_penalty,
            top_k=self.top_k,
            top_p=self.top_p,
            enable_text_splitting=self.enable_text_splitting,
        )
        wav = result["wav"] if isinstance(result, dict) else result
        if hasattr(wav, "detach"):
            wav = wav.detach().cpu().numpy()
        return np.asarray(wav, dtype=np.float32).reshape(-1)
