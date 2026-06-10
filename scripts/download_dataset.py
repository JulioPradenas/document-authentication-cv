#!/usr/bin/env python3
"""
Download MIDV-500 dataset for document authentication project.

Usage:
    python scripts/download_dataset.py
    python scripts/download_dataset.py --output-dir data/raw/midv500
"""

import argparse
import os
import sys
from pathlib import Path


def count_files(directory: Path) -> tuple[int, int]:
    """Return (file_count, total_bytes) for all files under directory."""
    total_files = 0
    total_bytes = 0
    for f in directory.rglob("*"):
        if f.is_file():
            total_files += 1
            total_bytes += f.stat().st_size
    return total_files, total_bytes


def download_via_midv500_package(output_dir: Path) -> bool:
    """Attempt download using the midv500 Python package. Returns True on success."""
    try:
        import midv500  # type: ignore[import]
    except ImportError:
        print("midv500 package not available, falling back to direct download.")
        return False

    print("Using midv500 package to download dataset...")
    try:
        midv500.download(str(output_dir))
        return True
    except Exception as exc:
        print(f"midv500 package download failed: {exc}")
        return False


def download_direct(output_dir: Path) -> bool:
    """
    Fall back: download MIDV-500 via direct HTTP from the official source.

    MIDV-500 is hosted at:
      http://l3i-share.univ-lr.fr/MIDV500/midv500_video.zip  (video clips)
    Subsets are available at ftp.l3i.univ-lr.fr — the zip is ~8 GB.

    For CI / quick smoke-tests the script accepts an env var
    MIDV500_URL to override the download URL.
    """
    import urllib.request
    import zipfile

    url = os.environ.get(
        "MIDV500_URL",
        "http://l3i-share.univ-lr.fr/MIDV500/midv500_video.zip",
    )
    zip_path = output_dir / "midv500_video.zip"

    print(f"Downloading MIDV-500 from {url} ...")
    print("Note: the full archive is ~8 GB — this may take a while.")

    try:
        def _progress(block_num: int, block_size: int, total_size: int) -> None:
            downloaded = block_num * block_size
            if total_size > 0:
                pct = min(100.0, downloaded / total_size * 100)
                print(f"\r  {pct:.1f}%  ({downloaded // 1_048_576} MB)", end="", flush=True)

        urllib.request.urlretrieve(url, zip_path, reporthook=_progress)
        print()  # newline after progress
    except Exception as exc:
        print(f"\nDownload failed: {exc}")
        return False

    print(f"Extracting {zip_path} ...")
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(output_dir)
        zip_path.unlink()
        print("Extraction complete, zip removed.")
    except Exception as exc:
        print(f"Extraction failed: {exc}")
        return False

    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download MIDV-500 dataset for document authentication project."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw/midv500"),
        help="Destination directory (default: data/raw/midv500)",
    )
    args = parser.parse_args()

    output_dir: Path = args.output_dir

    # Idempotency check — skip if already populated
    if output_dir.exists() and any(output_dir.iterdir()):
        file_count, total_bytes = count_files(output_dir)
        if file_count > 0:
            print(f"Dataset already present at {output_dir} ({file_count} files). Skipping download.")
            _print_summary(output_dir, file_count, total_bytes)
            return

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Target directory: {output_dir.resolve()}")

    # Try package first, then direct HTTP
    success = download_via_midv500_package(output_dir)
    if not success:
        success = download_direct(output_dir)

    if not success:
        print(
            "\nAll download methods failed.\n"
            "You can manually download MIDV-500 from:\n"
            "  https://github.com/fcakyon/midv500\n"
            "  http://l3i-share.univ-lr.fr/MIDV500/\n"
            f"and place the extracted files in: {output_dir.resolve()}"
        )
        sys.exit(1)

    file_count, total_bytes = count_files(output_dir)
    print(f"\nDownload complete.")
    _print_summary(output_dir, file_count, total_bytes)


def _print_summary(output_dir: Path, file_count: int, total_bytes: int) -> None:
    size_mb = total_bytes / 1_048_576
    print(f"\n--- Summary ---")
    print(f"  Location : {output_dir.resolve()}")
    print(f"  Files    : {file_count:,}")
    print(f"  Size     : {size_mb:.1f} MB")

    # Show up to 5 example paths
    examples = [f for f in output_dir.rglob("*") if f.is_file()][:5]
    if examples:
        print("  Examples :")
        for p in examples:
            print(f"    {p}")


if __name__ == "__main__":
    main()
