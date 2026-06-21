"""TTS Dataset Cleaner — neural audio cleaning for TTS training data."""

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
