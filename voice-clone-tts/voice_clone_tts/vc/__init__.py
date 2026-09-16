"""Voice conversion: train a model on a speaker, then recolour synthesized speech.

Importing this package registers the built-in converters. Like the TTS
backends, they only touch their external toolchains when actually used.
"""

from voice_clone_tts.vc.base import (
    VoiceConversionError,
    VoiceConverter,
    VoiceModel,
    converter_class,
    get_converter,
    iter_converter_info,
    list_converters,
    register_converter,
)
from voice_clone_tts.vc.dataset import TrainingDataset, build_training_dataset
from voice_clone_tts.vc.runner import ToolchainError

# Import for the side effect of registering the converters.
from voice_clone_tts.vc import rvc as _rvc  # noqa: F401,E402
from voice_clone_tts.vc import sovits as _sovits  # noqa: F401,E402

DEFAULT_CONVERTER = "rvc"

__all__ = [
    "DEFAULT_CONVERTER",
    "ToolchainError",
    "TrainingDataset",
    "VoiceConversionError",
    "VoiceConverter",
    "VoiceModel",
    "build_training_dataset",
    "converter_class",
    "get_converter",
    "iter_converter_info",
    "list_converters",
    "register_converter",
]
