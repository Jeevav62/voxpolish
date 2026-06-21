"""VoxPolish — TTS Dataset Cleaner · Gradio UI"""

import os
import io
import time
import zipfile
import tempfile
from pathlib import Path

import gradio as gr
import torch

from cleaner.enhance import load_model
from cleaner.dataset_io import detect_dataset_type, discover_audio_files, output_path_for, copy_metadata
from cleaner.pipeline import process_file, PipelineConfig

# Load model at startup
load_model(post_filter=True)


def process_zip(
    zip_file,
    use_pf: bool,
    do_trim: bool,
    do_norm: bool,
    target_lufs: float,
    atten_lim: float,
    progress=gr.Progress(),
):
    if zip_file is None:
        return None, "Upload a ZIP file first."

    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir  = Path(tmpdir) / "input"
        output_dir = Path(tmpdir) / "output"
        input_dir.mkdir()
        output_dir.mkdir()

        # Extract ZIP
        with zipfile.ZipFile(zip_file, "r") as z:
            z.extractall(input_dir)

        # Flatten one extra nesting level if zip had a single root folder
        contents = list(input_dir.iterdir())
        if len(contents) == 1 and contents[0].is_dir():
            input_dir = contents[0]

        dtype = detect_dataset_type(input_dir)
        files = discover_audio_files(input_dir, dtype)

        if not files:
            return None, "No audio files found in ZIP."

        copy_metadata(input_dir, output_dir, dtype)

        cfg = PipelineConfig(
            post_filter  = use_pf,
            atten_lim_db = atten_lim if atten_lim > 0 else None,
            trim         = do_trim,
            normalize    = do_norm,
            target_lufs  = target_lufs,
        )

        results = []
        t_start = time.perf_counter()

        for i, f in enumerate(files):
            progress((i + 1) / len(files), desc=f"Processing {f.name}")
            out = output_path_for(f, input_dir, output_dir)
            results.append(process_file(f, out, cfg))

        total_s = time.perf_counter() - t_start
        ok      = [r for r in results if r.success]
        failed  = [r for r in results if not r.success]
        avg_ms  = sum(r.latency_ms for r in ok) / len(ok) if ok else 0

        vram_mb = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0

        # Pack output into ZIP
        out_zip_path = Path(tmpdir) / "cleaned_dataset.zip"
        with zipfile.ZipFile(out_zip_path, "w", zipfile.ZIP_DEFLATED) as zout:
            for f in output_dir.rglob("*"):
                if f.is_file():
                    zout.write(f, f.relative_to(output_dir))

        # Read zip bytes before tmpdir is deleted
        zip_bytes = out_zip_path.read_bytes()

    # Write to a persistent temp file for Gradio download
    out_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
    out_tmp.write(zip_bytes)
    out_tmp.close()

    stats = f"""
## Results

| | |
|---|---|
| **Dataset Type** | {dtype} |
| **Files Processed** | {len(ok)} / {len(files)} |
| **Failed** | {len(failed)} |
| **Avg Latency/file** | {avg_ms:.1f} ms |
| **Total Time** | {total_s:.1f} s |
| **VRAM Used** | {vram_mb:.1f} MB |
""".strip()

    if failed:
        stats += "\n\n**Errors:**\n" + "\n".join(f"- `{r.path}`: {r.error}" for r in failed[:5])

    return out_tmp.name, stats


# ── UI ───────────────────────────────────────────────────────────────────────
with gr.Blocks(title="VoxPolish — TTS Dataset Cleaner") as demo:
    gr.Markdown("""
# 🎙️ VoxPolish — TTS Dataset Cleaner
Upload your dataset as a **ZIP file** → get back a clean, training-ready dataset.

**Pipeline:** neural noise removal → silence trimming → loudness normalization (-23 LUFS)

Supports: raw audio folders, LJSpeech format (wavs/ + metadata.csv), VCTK format.
""")

    with gr.Row():
        with gr.Column(scale=1):
            zip_in = gr.File(label="Upload Dataset ZIP", file_types=[".zip"])

            gr.Markdown("### Pipeline Options")
            use_pf    = gr.Checkbox(label="Post-filter (extra noise suppression)", value=True)
            do_trim   = gr.Checkbox(label="Trim silence (start & end)", value=True)
            do_norm   = gr.Checkbox(label="Loudness normalization", value=True)
            target_lufs = gr.Slider(-35, -14, value=-23, step=0.5, label="Target LUFS (default: -23)")
            atten_lim = gr.Slider(0, 40, value=0, step=1, label="Attenuation limit dB (0 = no limit)")

            btn = gr.Button("Clean Dataset", variant="primary", size="lg")

        with gr.Column(scale=1):
            zip_out  = gr.File(label="Download Cleaned Dataset ZIP")
            stats_md = gr.Markdown()

    btn.click(
        fn=process_zip,
        inputs=[zip_in, use_pf, do_trim, do_norm, target_lufs, atten_lim],
        outputs=[zip_out, stats_md],
    )


if __name__ == "__main__":
    demo.launch(share=False, inbrowser=True, theme=gr.themes.Soft())
