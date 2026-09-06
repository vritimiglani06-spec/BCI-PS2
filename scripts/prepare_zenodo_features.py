from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


LABEL_MAP_COLUMNS = {
    "HVHA": "high_valence_high_arousal",
    "HVLA": "high_valence_low_arousal",
    "LVHA": "low_valence_high_arousal",
    "LVLA": "low_valence_low_arousal",
}


def infer_label(row) -> str:
    if "label" in row and pd.notna(row["label"]):
        return str(row["label"])
    if "valence" in row and "arousal" in row:
        v = "HV" if float(row["valence"]) > 5 else "LV"
        a = "HA" if float(row["arousal"]) > 5 else "LA"
        return v + a
    raise ValueError("Could not infer label. Provide a label column or valence/arousal columns.")


def normalize_feature_csv(input_csv: Path, output_csv: Path, label_col: str = "label"):
    df = pd.read_csv(input_csv)
    df[label_col] = df.apply(infer_label, axis=1)
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    keep = [label_col] + numeric_cols
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df[keep].to_csv(output_csv, index=False)


def main():
    parser = argparse.ArgumentParser(
        description="Normalize existing Zenodo feature CSV files into the manifest format expected by ASJDA."
    )
    parser.add_argument("--input-dir", required=True, help="Folder containing per-subject feature CSV files.")
    parser.add_argument("--output-dir", required=True, help="Folder where normalized CSV files and manifest are written.")
    parser.add_argument("--device", default="emotiv", help="Device label to store in the manifest.")
    parser.add_argument("--glob", default="*.csv")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    feature_dir = output_dir / "features"
    rows = []

    for path in sorted(input_dir.glob(args.glob)):
        subject_id = path.stem
        out_path = feature_dir / f"{subject_id}.csv"
        normalize_feature_csv(path, out_path)
        rows.append({
            "subject_id": subject_id,
            "device": args.device,
            "session": "1",
            "file": str(out_path.relative_to(output_dir)),
        })

    if not rows:
        raise SystemExit(f"No files matched {input_dir / args.glob}")

    pd.DataFrame(rows).to_csv(output_dir / "manifest.csv", index=False)
    print(f"wrote {output_dir / 'manifest.csv'}")


if __name__ == "__main__":
    main()
