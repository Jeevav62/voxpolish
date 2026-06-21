# Contributing to TTS Dataset Cleaner

Thanks for your interest in contributing! This project aims to be the simplest, most reliable way to clean audio datasets for TTS training. Every contribution helps.

## Ways to Contribute

- 🐛 **Report bugs** — open an issue with steps to reproduce
- 💡 **Suggest features** — open a feature request issue
- 📝 **Improve docs** — typos, clearer examples, better explanations
- 🔧 **Submit code** — fix a bug or add a feature via pull request

## Development Setup

```bash
git clone https://github.com/Jeevav62/voxpolish.git
cd voxpolish

python -m venv venv
# Windows: venv\Scripts\activate
# Linux/macOS: source venv/bin/activate

# Install PyTorch (GPU or CPU — see README)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

## Testing Your Changes

Before opening a PR, verify the pipeline still works end to end:

```bash
# Put a few short WAV files in samples/noisy/, then:
python clean.py --input ./samples/noisy --output ./samples/clean

# Confirm: files cleaned, report shows 0 failures, output loudness ~-23 LUFS
python -c "import soundfile as sf, pyloudnorm as pyln; d,r=sf.read('samples/clean/YOUR_FILE.wav'); print(pyln.Meter(r).integrated_loudness(d))"
```

Also smoke-test the web UI:
```bash
python app.py
```

## Pull Request Guidelines

1. **One change per PR** — keep it focused and easy to review
2. **Describe what and why** — what does this fix/add, and why
3. **Match the existing style** — the code is plain, readable Python; keep it that way
4. **Don't commit** — virtual environments, model checkpoints, audio datasets, or `__pycache__`
5. **Update docs** — if you change behavior or add a flag, update the README

## Ideas We'd Love Help With

- Additional dataset formats (Common Voice, custom layouts)
- Voice Activity Detection (VAD) to drop non-speech clips
- Per-file quality scoring (SNR / MOS estimation)
- Batched GPU inference for higher throughput on large datasets
- Resumable runs (skip already-cleaned files)

## Code of Conduct

Be kind, be constructive, assume good faith. We're here to build something useful together.

## Questions?

Open a [Discussion](https://github.com/Jeevav62/voxpolish/discussions) or an issue. Happy to help.
