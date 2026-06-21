"""
TTS Dataset Cleaner — CLI

Usage:
    python clean.py --input ./my_dataset --output ./my_dataset_clean

Options:
    --input         Input folder (raw audio or LJSpeech/VCTK dataset)
    --output        Output folder (created if not exists)
    --no-pf         Disable post-filter (default: enabled)
    --atten-lim     Noise attenuation limit in dB (e.g. 12). Default: no limit
    --no-trim       Skip silence trimming
    --no-norm       Skip loudness normalization
    --target-lufs   Loudness target in LUFS (default: -23.0)
    --workers       Parallel workers (default: 1, GPU needs 1)
"""

import argparse
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from cleaner.enhance import load_model
from cleaner.dataset_io import detect_dataset_type, discover_audio_files, output_path_for, copy_metadata
from cleaner.pipeline import process_file, PipelineConfig, FileResult


def parse_args():
    p = argparse.ArgumentParser(description="TTS Dataset Cleaner")
    p.add_argument("--input",       required=True,  type=Path)
    p.add_argument("--output",      required=True,  type=Path)
    p.add_argument("--no-pf",       action="store_true", help="Disable post-filter")
    p.add_argument("--atten-lim",   type=float, default=None)
    p.add_argument("--no-trim",     action="store_true")
    p.add_argument("--no-norm",     action="store_true")
    p.add_argument("--target-lufs", type=float, default=-23.0)
    p.add_argument("--workers",     type=int,   default=1)
    return p.parse_args()


def write_report(output_dir: Path, results: list[FileResult], dataset_type: str):
    ok     = [r for r in results if r.success]
    failed = [r for r in results if not r.success]
    avg_ms = sum(r.latency_ms for r in ok) / len(ok) if ok else 0

    lines = [
        "TTS Dataset Cleaner - Report",
        "=" * 40,
        f"Dataset type   : {dataset_type}",
        f"Total files    : {len(results)}",
        f"Cleaned        : {len(ok)}",
        f"Failed         : {len(failed)}",
        f"Avg latency    : {avg_ms:.1f} ms/file",
        "",
    ]
    if failed:
        lines.append("Failed files:")
        for r in failed:
            lines.append(f"  {r.path}: {r.error}")

    report_path = output_dir / "cleaning_report.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nReport saved: {report_path}")


def main():
    args = parse_args()

    if not args.input.exists():
        print(f"Error: --input '{args.input}' does not exist.")
        sys.exit(1)

    args.output.mkdir(parents=True, exist_ok=True)

    # Detect dataset type
    dtype = detect_dataset_type(args.input)
    print(f"Dataset type: {dtype}")

    # Load model once
    load_model(post_filter=not args.no_pf)

    # Discover files
    files = discover_audio_files(args.input, dtype)
    if not files:
        print("No audio files found.")
        sys.exit(1)
    print(f"Found {len(files)} audio files.")

    # Copy metadata unchanged
    copy_metadata(args.input, args.output, dtype)

    cfg = PipelineConfig(
        post_filter   = not args.no_pf,
        atten_lim_db  = args.atten_lim,
        trim          = not args.no_trim,
        normalize     = not args.no_norm,
        target_lufs   = args.target_lufs,
    )

    results = []

    def _process(f):
        out = output_path_for(f, args.input, args.output)
        return process_file(f, out, cfg)

    if args.workers == 1:
        for f in tqdm(files, desc="Cleaning"):
            results.append(_process(f))
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(_process, f): f for f in files}
            for fut in tqdm(as_completed(futures), total=len(files), desc="Cleaning"):
                results.append(fut.result())

    write_report(args.output, results, dtype)


if __name__ == "__main__":
    main()
