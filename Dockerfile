# VoxPolish — TTS Dataset Cleaner
# CPU image. Builds the Gradio web UI on port 7860.
FROM python:3.10-slim

# System deps: ffmpeg + libsndfile for audio I/O
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Pin torch/torchaudio to 2.3.0 CPU (the version the denoiser is verified against)
RUN pip install --no-cache-dir \
        torch==2.3.0 torchaudio==2.3.0 \
        --index-url https://download.pytorch.org/whl/cpu

# App dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY . .

# Gradio web UI
EXPOSE 7860
ENV GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    GRADIO_INBROWSER=0

CMD ["python", "app.py"]
