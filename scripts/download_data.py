"""Download the MetroPT-3 dataset from the UCI Machine Learning Repository.

The ZIP archive (~208 MB) is saved to ``data/raw/`` and the CSV
``MetroPT3(AirCompressor).csv`` is extracted next to it. Existing files are
reused unless ``--force`` is given.

Dataset: Davari, N., Veloso, B., Ribeiro, R., & Gama, J. (2021).
MetroPT-3 Dataset. UCI Machine Learning Repository.
https://doi.org/10.24432/C5VW3R (CC BY 4.0).

Usage::

    python scripts/download_data.py            # download + extract
    python scripts/download_data.py --force    # re-download
"""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

import requests

# Make ``src`` importable when the script is run from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402

CHUNK_SIZE = 1 << 20  # 1 MiB


def download_file(url: str, target: Path, timeout: int = 60) -> Path:
    """Stream ``url`` to ``target`` and show simple progress.

    The file is written to ``<target>.part`` first and renamed on success,
    so an interrupted download never leaves a corrupt archive behind.

    Args:
        url: Source URL.
        target: Destination file path.
        timeout: Connection/read timeout in seconds.

    Returns:
        Path of the downloaded file.

    Raises:
        requests.HTTPError: If the server returns an error status.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    with requests.get(url, stream=True, timeout=timeout) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        done = 0
        with open(tmp, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                fh.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r  {done / 1e6:7.1f} / {total / 1e6:.1f} MB "
                          f"({100 * done / total:5.1f} %)", end="", flush=True)
                else:
                    print(f"\r  {done / 1e6:7.1f} MB", end="", flush=True)
    print()
    tmp.replace(target)
    return target


def extract_csv(zip_path: Path, csv_name: str, out_dir: Path) -> Path:
    """Extract ``csv_name`` from ``zip_path`` into ``out_dir``.

    The CSV may sit in a sub-folder of the archive; it is written directly
    to ``out_dir/csv_name``. Only this member is extracted.

    Args:
        zip_path: Path of the ZIP archive.
        csv_name: File name of the CSV inside the archive.
        out_dir: Destination directory.

    Returns:
        Path of the extracted CSV.

    Raises:
        FileNotFoundError: If the archive does not contain ``csv_name``.
    """
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.namelist() if Path(m).name == csv_name]
        if not members:
            raise FileNotFoundError(f"{csv_name} not in archive. Members: {zf.namelist()}")
        out_path = out_dir / csv_name
        with zf.open(members[0]) as src, open(out_path, "wb") as dst:
            shutil.copyfileobj(src, dst)
    return out_path


def count_rows(csv_path: Path) -> int:
    """Count data rows (lines minus header) without loading the CSV."""
    with open(csv_path, "rb") as fh:
        return sum(1 for _ in fh) - 1


def main(argv: list[str] | None = None) -> int:
    """Download and extract MetroPT-3; return a process exit code."""
    parser = argparse.ArgumentParser(description="Download MetroPT-3 from UCI.")
    parser.add_argument("--url", default=config.DATASET_URL, help="Archive URL.")
    parser.add_argument("--out-dir", type=Path, default=config.RAW_DATA_DIR)
    parser.add_argument("--force", action="store_true", help="Re-download even if present.")
    parser.add_argument("--keep-zip", action="store_true", help="Keep the ZIP after extraction.")
    args = parser.parse_args(argv)

    csv_path = args.out_dir / config.RAW_CSV_NAME
    zip_path = args.out_dir / config.DATASET_ZIP_NAME

    if csv_path.exists() and not args.force:
        print(f"CSV already present: {csv_path} (use --force to re-download)")
    else:
        if not zip_path.exists() or args.force:
            print(f"Downloading {args.url}")
            download_file(args.url, zip_path)
        print(f"Extracting {config.RAW_CSV_NAME}")
        extract_csv(zip_path, config.RAW_CSV_NAME, args.out_dir)
        if not args.keep_zip:
            zip_path.unlink(missing_ok=True)

    n_rows = count_rows(csv_path)
    print(f"Rows in CSV: {n_rows:,} (UCI states {config.EXPECTED_N_ROWS:,})")
    if n_rows != config.EXPECTED_N_ROWS:
        print("WARNING: row count differs from the UCI description.", file=sys.stderr)
    print(f"Done: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
