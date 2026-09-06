from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile


def main():
    parser = argparse.ArgumentParser(description="Download DEAP from Kaggle using configured Kaggle credentials.")
    parser.add_argument("--output-dir", default="outputs/asjda_implementation/raw/deap_kaggle")
    parser.add_argument("--dataset", default="manh123df/deap-dataset")
    parser.add_argument("--unzip", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "kaggle",
        "datasets",
        "download",
        "-d",
        args.dataset,
        "-p",
        str(output_dir),
    ]
    if args.unzip:
        cmd.append("--unzip")
    subprocess.run(cmd, check=True)

    if not args.unzip:
        for zip_path in output_dir.glob("*.zip"):
            with ZipFile(zip_path) as zf:
                zf.extractall(output_dir / zip_path.stem)
            print(f"extracted {zip_path}")

    print(f"downloaded to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
