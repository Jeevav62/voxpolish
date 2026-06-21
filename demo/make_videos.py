"""Build before->after demo MP4s (waveform frame + audio) for inline README playback."""
import subprocess
import numpy as np
import soundfile as sf
import torchaudio
import torch
from pathlib import Path

TARGET_SR = 48000
GAP_S = 0.4  # silence gap between before and after

SAMPLES = [
    ("jeeva_before.ogg", "jeeva_after.wav", "waveform_jeeva.png", "jeeva_demo.mp4"),
    ("car_before.wav",   "car_after.wav",   "waveform_car.png",   "car_demo.mp4"),
]


def load_48k_mono(path):
    d, r = sf.read(path)
    if d.ndim > 1:
        d = d.mean(axis=1)
    t = torch.from_numpy(d.astype(np.float32)).unsqueeze(0)
    if r != TARGET_SR:
        t = torchaudio.functional.resample(t, r, TARGET_SR)
    return t.squeeze(0).numpy()


for before, after, frame, out in SAMPLES:
    yb = load_48k_mono(before)
    ya = load_48k_mono(after)
    gap = np.zeros(int(TARGET_SR * GAP_S), dtype=np.float32)
    combined = np.concatenate([yb, gap, ya])
    # normalize peak to avoid clipping in the demo
    peak = np.abs(combined).max()
    if peak > 0:
        combined = combined / peak * 0.95
    combo_wav = out.replace(".mp4", "_combined.wav")
    sf.write(combo_wav, combined, TARGET_SR)

    cmd = [
        "ffmpeg", "-y",
        "-loop", "1", "-i", frame,
        "-i", combo_wav,
        "-c:v", "libx264", "-tune", "stillimage", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest", out,
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    Path(combo_wav).unlink()
    print(f"built {out}")
