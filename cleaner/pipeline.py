import time
import torch
import soundfile as sf
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from df.io import load_audio, save_audio
from cleaner.enhance import enhance_audio, sample_rate
from cleaner.audio_utils import trim_silence, normalize_loudness, tensor_to_numpy, numpy_to_tensor

MODEL_SR = 48000  # denoiser model requires 48kHz


@dataclass
class FileResult:
    path: str
    success: bool
    latency_ms: float = 0.0
    error: str = ""


@dataclass
class PipelineConfig:
    post_filter: bool     = True
    atten_lim_db: Optional[float] = None
    trim: bool            = True
    normalize: bool       = True
    target_lufs: float    = -23.0


def process_file(
    input_path: Path,
    output_path: Path,
    cfg: PipelineConfig,
) -> FileResult:
    t0 = time.perf_counter()
    try:
        # 1. Load + resample to 48kHz for the denoiser model
        audio, meta = load_audio(str(input_path), sr=MODEL_SR)
        orig_sr = meta.sample_rate

        # 2. Neural noise removal
        enhanced = enhance_audio(audio, atten_lim_db=cfg.atten_lim_db)

        # 3. Resample back to original SR before post-processing
        if orig_sr != MODEL_SR:
            import torchaudio
            enhanced = torchaudio.functional.resample(enhanced, MODEL_SR, orig_sr)

        audio_np = tensor_to_numpy(enhanced)

        # 4. Silence trimming
        if cfg.trim:
            audio_np = trim_silence(audio_np, orig_sr)

        # 5. Loudness normalization
        if cfg.normalize:
            audio_np = normalize_loudness(audio_np, orig_sr, cfg.target_lufs)

        # 6. Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out_audio = numpy_to_tensor(audio_np)
        save_audio(str(output_path), out_audio, orig_sr)

        latency_ms = (time.perf_counter() - t0) * 1000
        return FileResult(path=str(input_path), success=True, latency_ms=latency_ms)

    except Exception as e:
        return FileResult(path=str(input_path), success=False, error=str(e))
