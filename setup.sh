#!/usr/bin/env bash
# VoxPolish — one-command setup
# Creates a virtual environment, installs PyTorch (GPU if available, else CPU),
# and all dependencies. Works on Linux, macOS, and Windows (Git Bash / WSL).
set -e

echo "==> Setting up VoxPolish"

# 1. Virtual environment
if [ ! -d venv ]; then
  python -m venv venv 2>/dev/null || python3 -m venv venv
fi

# 2. Activate (handle Unix vs Windows Git Bash layouts)
if [ -f venv/bin/activate ]; then
  source venv/bin/activate
else
  source venv/Scripts/activate
fi

python -m pip install --upgrade pip

# 3. PyTorch — CUDA build if an NVIDIA GPU is present, otherwise CPU
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "==> NVIDIA GPU detected — installing CUDA PyTorch"
  pip install torch==2.3.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
else
  echo "==> No GPU detected — installing CPU PyTorch"
  pip install torch==2.3.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cpu
fi

# 4. Everything else
pip install -r requirements.txt

echo ""
echo "==> Done. Activate the venv, then run:"
echo "    Web UI :  python app.py"
echo "    CLI    :  python clean.py --input ./my_dataset --output ./my_dataset_clean"
