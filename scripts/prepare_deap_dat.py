from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from asjda.data import load_deap_dat_folder
from asjda.features import extract_de_features, repeat_trial_labels, segment_trials


def main():
    parser = argparse.ArgumentParser(
        description="Convert official/Kaggle DEAP .dat subject files into ASJDA feature CSVs."
    )
    parser.add_argument("--deap-dir", required=True, help="Folder containing s01.dat ... s32.dat.")
    parser.add_argument("--output-dir", required=True, help="Output folder for manifest.csv and feature CSVs.")
    parser.add_argument("--label-mode", choices=["valence", "arousal"], default="valence")
    parser.add_argument("--sfreq", type=float, default=128.0)
    parser.add_argument("--window-seconds", type=float, default=4.0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    feature_dir = output_dir / f"features_deap_{args.label_mode}"
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_dir.mkdir(parents=True, exist_ok=True)

    subjects = load_deap_dat_folder(args.deap_dir, args.label_mode)
    manifest_rows = []
    for subject_id, subject in sorted(subjects.items()):
        segments = segment_trials(subject.x, args.sfreq, args.window_seconds)
        segments_per_trial = segments.shape[0] // len(subject.y)
        labels = repeat_trial_labels(subject.y, segments_per_trial)
        features, names = extract_de_features(segments, args.sfreq)

        df = pd.DataFrame(features, columns=names)
        df.insert(0, "label", labels)
        df["label"] = df["label"].map({0: "low", 1: "high"})

        out_file = feature_dir / f"{subject_id}.csv"
        df.to_csv(out_file, index=False)
        manifest_rows.append({
            "subject_id": subject_id,
            "device": "deap_biosemi",
            "session": "1",
            "label_mode": args.label_mode,
            "file": str(out_file.relative_to(output_dir)),
        })

    manifest_path = output_dir / f"manifest_deap_{args.label_mode}.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    print(f"wrote {manifest_path}")
    print(f"subjects={len(manifest_rows)}")


if __name__ == "__main__":
    main()
