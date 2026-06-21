---
title: VoxPolish TTS Dataset Cleaner
emoji: 🎙️
colorFrom: indigo
colorTo: green
sdk: gradio
sdk_version: 5.9.1
python_version: "3.10"
app_file: app.py
pinned: true
license: mit
short_description: Clean noisy audio for TTS training data
---

# 🎙️ VoxPolish — TTS Dataset Cleaner

Upload noisy speech and hear it cleaned instantly. Powered by neural denoising, with silence trimming and loudness normalization to -23 LUFS — the same pipeline used to prepare datasets for text-to-speech training.

**Two modes:**
- **Quick Clean** — drop a single audio file, hear the cleaned result + see latency/RTF stats.
- **Dataset Cleaner** — upload a ZIP of your dataset, download the cleaned version.

🔗 **Full project + CLI:** [github.com/Jeevav62/voxpolish](https://github.com/Jeevav62/voxpolish)

Built by [Jeeva](https://huggingface.co/jeevav62) — I turn research papers into shipping products (TTS/LLM fine-tuning, audio ML tooling). Open to collaboration and hiring.
