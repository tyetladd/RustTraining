"""TTS backends.

Importing this package registers every built-in backend. The modules only
import their heavy dependencies inside :meth:`TTSBackend.load`, so this stays
cheap even without torch installed.
"""

from voice_clone_tts.backends.base import (
    SynthesisRequest,
    TTSBackend,
    available_backends,
    backend_class,
    get_backend,
    iter_backend_info,
    list_backends,
    register_backend,
)

# Import for the side effect of registering the backends.
from voice_clone_tts.backends import dummy as _dummy  # noqa: F401,E402
from voice_clone_tts.backends import silero as _silero  # noqa: F401,E402
from voice_clone_tts.backends import xtts as _xtts  # noqa: F401,E402

DEFAULT_BACKEND = "xtts"

__all__ = [
    "DEFAULT_BACKEND",
    "SynthesisRequest",
    "TTSBackend",
    "available_backends",
    "backend_class",
    "get_backend",
    "iter_backend_info",
    "list_backends",
    "register_backend",
]
