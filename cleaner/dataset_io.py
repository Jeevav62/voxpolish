import os
import shutil
from pathlib import Path
from typing import List, Tuple

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}


def detect_dataset_type(input_dir: Path) -> str:
    """Returns 'ljspeech', 'vctk', or 'raw_folder'."""
    if (input_dir / "metadata.csv").exists() or (input_dir / "metadata.txt").exists():
        return "ljspeech"
    subdirs = [d for d in input_dir.iterdir() if d.is_dir()]
    if subdirs and all(
        any((d / sub).exists() for sub in ["wav", "wav48", "wav16"])
        for d in subdirs[:3]
    ):
        return "vctk"
    return "raw_folder"


def discover_audio_files(input_dir: Path, dataset_type: str) -> List[Path]:
    """Return list of audio file paths to process."""
    if dataset_type == "ljspeech":
        wav_dir = input_dir / "wavs"
        if not wav_dir.exists():
            wav_dir = input_dir
        return sorted([f for f in wav_dir.rglob("*") if f.suffix.lower() in AUDIO_EXTENSIONS])

    if dataset_type == "vctk":
        files = []
        for spk_dir in sorted(input_dir.iterdir()):
            if not spk_dir.is_dir():
                continue
            for wav_sub in ["wav48", "wav", "wav16"]:
                wav_path = spk_dir / wav_sub
                if wav_path.exists():
                    files.extend(sorted([f for f in wav_path.rglob("*") if f.suffix.lower() in AUDIO_EXTENSIONS]))
                    break
        return files

    # raw_folder — recurse everything
    return sorted([f for f in input_dir.rglob("*") if f.suffix.lower() in AUDIO_EXTENSIONS])


def output_path_for(input_file: Path, input_dir: Path, output_dir: Path) -> Path:
    """Mirror input_file's relative path under output_dir."""
    rel = input_file.relative_to(input_dir)
    out = output_dir / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    return out.with_suffix(".wav")


def copy_metadata(input_dir: Path, output_dir: Path, dataset_type: str):
    """Copy metadata files (CSV, txt) unchanged to output dir."""
    if dataset_type == "ljspeech":
        for name in ["metadata.csv", "metadata.txt"]:
            src = input_dir / name
            if src.exists():
                shutil.copy2(src, output_dir / name)
