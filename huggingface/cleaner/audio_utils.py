import numpy as np
import torch
import pyloudnorm as pyln
import soundfile as sf
from typing import Tuple

SILENCE_DB   = -50.0   # dB below which a frame is considered silence
TOP_DB       = 60.0    # torchaudio trim threshold
TARGET_LUFS  = -23.0   # EBU R128 — standard for TTS training data


def trim_silence(audio: np.ndarray, sr: int, top_db: float = TOP_DB) -> np.ndarray:
    """Trim leading and trailing silence from a numpy audio array [T] or [C, T]."""
    mono = audio if audio.ndim == 1 else audio.mean(axis=0)
    frame_len = int(sr * 0.025)   # 25ms frames
    hop_len   = int(sr * 0.010)   # 10ms hop

    energy = np.array([
        np.sqrt(np.mean(mono[i:i+frame_len]**2) + 1e-10)
        for i in range(0, len(mono) - frame_len, hop_len)
    ])
    db = 20 * np.log10(energy + 1e-10)
    speech = db > (db.max() - top_db)

    if not speech.any():
        return audio

    start_frame = np.argmax(speech)
    end_frame   = len(speech) - np.argmax(speech[::-1])
    start_sample = max(0, start_frame * hop_len - frame_len)
    end_sample   = min(audio.shape[-1], end_frame * hop_len + frame_len)

    if audio.ndim == 1:
        return audio[start_sample:end_sample]
    return audio[:, start_sample:end_sample]


def normalize_loudness(audio: np.ndarray, sr: int, target_lufs: float = TARGET_LUFS) -> np.ndarray:
    """Normalize audio to target LUFS. Returns numpy array same shape."""
    meter = pyln.Meter(sr)
    # pyloudnorm expects [T] or [T, C]
    inp = audio.T if audio.ndim == 2 else audio
    loudness = meter.integrated_loudness(inp)
    if np.isinf(loudness) or np.isnan(loudness):
        return audio  # silent/near-silent — skip norm
    normalized = pyln.normalize.loudness(inp, loudness, target_lufs)
    # clip to prevent clipping
    normalized = np.clip(normalized, -1.0, 1.0)
    return normalized.T if audio.ndim == 2 else normalized


def tensor_to_numpy(audio: torch.Tensor) -> np.ndarray:
    """[C, T] tensor → numpy [C, T] float32"""
    return audio.squeeze().cpu().numpy().astype(np.float32)


def numpy_to_tensor(audio: np.ndarray) -> torch.Tensor:
    """numpy [T] or [C, T] → [1, T] tensor"""
    if audio.ndim == 1:
        return torch.from_numpy(audio).unsqueeze(0)
    return torch.from_numpy(audio)
