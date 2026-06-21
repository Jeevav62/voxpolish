"""VoxPolish — TTS Dataset Cleaner. Neural audio cleaning for TTS training data."""

# Compat shim: some torchaudio versions removed torchaudio.backend.common.AudioMetaData,
# which the df library imports. Recreate the path if it's missing.
import sys as _sys
import types as _types
try:
    import torchaudio as _ta
    try:
        from torchaudio.backend.common import AudioMetaData as _AMD  # noqa: F401
    except Exception:
        _amd = getattr(_ta, "AudioMetaData", None)
        if _amd is not None:
            _backend = _sys.modules.get("torchaudio.backend") or _types.ModuleType("torchaudio.backend")
            _common = _types.ModuleType("torchaudio.backend.common")
            _common.AudioMetaData = _amd
            _backend.common = _common
            _sys.modules["torchaudio.backend"] = _backend
            _sys.modules["torchaudio.backend.common"] = _common
except Exception:
    pass

# Silence noisy third-party warnings BEFORE importing df / torch
import warnings

warnings.filterwarnings("ignore", message=".*torchaudio.backend.common.*")
warnings.filterwarnings("ignore", message=".*sinc_interpolation.*")
warnings.filterwarnings("ignore", message=".*cudnnException.*")
warnings.filterwarnings("ignore", message=".*Plan failed with a cudnnException.*")
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")

from cleaner.pipeline import process_file, PipelineConfig, FileResult  # noqa: E402
from cleaner.enhance import load_model  # noqa: E402

__version__ = "0.1.0"
__all__ = ["process_file", "PipelineConfig", "FileResult", "load_model"]
