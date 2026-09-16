"""Clone a voice from an audio sample and read arbitrary text with it.

Public entry points::

    from voice_clone_tts import synthesize, build_profile
    result = synthesize("sample.mp3", "Привет, мир!", out_path="out.wav")
"""

from voice_clone_tts.errors import (
    AudioError,
    BackendError,
    LanguageError,
    MissingDependencyError,
    VoiceCloneError,
)
from voice_clone_tts.config import PipelineConfig
from voice_clone_tts.profile import SpeakerProfile, build_profile
from voice_clone_tts.pipeline import SynthesisResult, synthesize

__version__ = "0.1.0"

__all__ = [
    "AudioError",
    "BackendError",
    "LanguageError",
    "MissingDependencyError",
    "PipelineConfig",
    "SpeakerProfile",
    "SynthesisResult",
    "VoiceCloneError",
    "build_profile",
    "synthesize",
    "__version__",
]
