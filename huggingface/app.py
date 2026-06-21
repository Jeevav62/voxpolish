"""VoxPolish — TTS Dataset Cleaner · HuggingFace Space"""

import time
import tempfile
import zipfile
from pathlib import Path

import gradio as gr
import torch
import numpy as np

from cleaner.enhance import load_model, enhance_audio, sample_rate
from cleaner.audio_utils import trim_silence, normalize_loudness, tensor_to_numpy, numpy_to_tensor
from cleaner.dataset_io import detect_dataset_type, discover_audio_files, output_path_for, copy_metadata
from cleaner.pipeline import process_file, PipelineConfig
from df.io import load_audio, save_audio

# Load model once (CPU on HF free tier, GPU if available)
load_model(post_filter=True)
MODEL_SR = 48000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── Quick Clean: single file ─────────────────────────────────────────────────
def quick_clean(audio_path, do_trim, do_norm, target_lufs):
    if audio_path is None:
        return None, "Upload an audio file first."

    audio, meta = load_audio(audio_path, sr=MODEL_SR)
    orig_sr = meta.sample_rate
    duration_s = audio.shape[-1] / MODEL_SR

    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    enhanced = enhance_audio(audio)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - t0) * 1000

    if orig_sr != MODEL_SR:
        import torchaudio
        enhanced = torchaudio.functional.resample(enhanced, MODEL_SR, orig_sr)

    audio_np = tensor_to_numpy(enhanced)
    if do_trim:
        audio_np = trim_silence(audio_np, orig_sr)
    if do_norm:
        audio_np = normalize_loudness(audio_np, orig_sr, target_lufs)

    out_path = tempfile.mktemp(suffix="_cleaned.wav")
    save_audio(out_path, numpy_to_tensor(audio_np), orig_sr)

    rtf = (latency_ms / 1000) / duration_s if duration_s > 0 else 0
    vram = f"{torch.cuda.max_memory_allocated()/1e6:.1f} MB" if DEVICE == "cuda" else "n/a (CPU)"

    stats = f"""
## Stats

| | |
|---|---|
| **Device** | {DEVICE.upper()} |
| **Latency** | {latency_ms:.0f} ms |
| **RTF** | {rtf:.3f}x {'✅ faster than real-time' if rtf < 1 else ''} |
| **Output sample rate** | {orig_sr} Hz |
| **VRAM** | {vram} |
""".strip()
    return out_path, stats


# ── Dataset Cleaner: ZIP ──────────────────────────────────────────────────────
def clean_zip(zip_file, do_trim, do_norm, target_lufs, progress=gr.Progress()):
    if zip_file is None:
        return None, "Upload a ZIP file first."

    with tempfile.TemporaryDirectory() as tmp:
        in_dir = Path(tmp) / "in"
        out_dir = Path(tmp) / "out"
        in_dir.mkdir(); out_dir.mkdir()

        with zipfile.ZipFile(zip_file, "r") as z:
            z.extractall(in_dir)
        contents = list(in_dir.iterdir())
        if len(contents) == 1 and contents[0].is_dir():
            in_dir = contents[0]

        dtype = detect_dataset_type(in_dir)
        files = discover_audio_files(in_dir, dtype)
        if not files:
            return None, "No audio files found in ZIP."

        copy_metadata(in_dir, out_dir, dtype)
        cfg = PipelineConfig(trim=do_trim, normalize=do_norm, target_lufs=target_lufs)

        results = []
        for i, f in enumerate(files):
            progress((i + 1) / len(files), desc=f"Cleaning {f.name}")
            results.append(process_file(f, output_path_for(f, in_dir, out_dir), cfg))

        ok = [r for r in results if r.success]
        avg_ms = sum(r.latency_ms for r in ok) / len(ok) if ok else 0

        zip_bytes_path = Path(tmp) / "cleaned.zip"
        with zipfile.ZipFile(zip_bytes_path, "w", zipfile.ZIP_DEFLATED) as zout:
            for f in out_dir.rglob("*"):
                if f.is_file():
                    zout.write(f, f.relative_to(out_dir))
        data = zip_bytes_path.read_bytes()

    out_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
    out_tmp.write(data); out_tmp.close()

    stats = f"""
## Results

| | |
|---|---|
| **Dataset type** | {dtype} |
| **Files cleaned** | {len(ok)} / {len(files)} |
| **Avg latency/file** | {avg_ms:.0f} ms |
""".strip()
    return out_tmp.name, stats


# ── UI ───────────────────────────────────────────────────────────────────────
with gr.Blocks(title="VoxPolish — TTS Dataset Cleaner", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
# 🎙️ VoxPolish — TTS Dataset Cleaner
Clean noisy speech for text-to-speech training: **neural denoise → silence trim → loudness normalize (-23 LUFS).**
[GitHub repo](https://github.com/Jeevav62/voxpolish) · built by [Jeeva](https://huggingface.co/jeevav62)
""")

    with gr.Tab("⚡ Quick Clean (single file)"):
        with gr.Row():
            with gr.Column():
                a_in = gr.Audio(label="Noisy audio", type="filepath", sources=["upload", "microphone"])
                q_trim = gr.Checkbox(label="Trim silence", value=True)
                q_norm = gr.Checkbox(label="Loudness normalize", value=True)
                q_lufs = gr.Slider(-35, -14, value=-23, step=0.5, label="Target LUFS")
                q_btn = gr.Button("Clean Audio", variant="primary", size="lg")
            with gr.Column():
                a_out = gr.Audio(label="Cleaned audio", type="filepath")
                q_stats = gr.Markdown()
        q_btn.click(quick_clean, [a_in, q_trim, q_norm, q_lufs], [a_out, q_stats])

    with gr.Tab("📦 Dataset Cleaner (ZIP)"):
        with gr.Row():
            with gr.Column():
                z_in = gr.File(label="Dataset ZIP", file_types=[".zip"])
                z_trim = gr.Checkbox(label="Trim silence", value=True)
                z_norm = gr.Checkbox(label="Loudness normalize", value=True)
                z_lufs = gr.Slider(-35, -14, value=-23, step=0.5, label="Target LUFS")
                z_btn = gr.Button("Clean Dataset", variant="primary", size="lg")
            with gr.Column():
                z_out = gr.File(label="Cleaned dataset ZIP")
                z_stats = gr.Markdown()
        z_btn.click(clean_zip, [z_in, z_trim, z_norm, z_lufs], [z_out, z_stats])


if __name__ == "__main__":
    demo.launch()
