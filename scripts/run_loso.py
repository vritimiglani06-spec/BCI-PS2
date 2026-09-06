from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from asjda.config import load_config
from asjda.data import load_manifest_dataset
from asjda.train import train_one_fold


def main():
    parser = argparse.ArgumentParser(description="Run ASJDA leave-one-subject-out evaluation.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)

    subjects, label_encoder, feature_cols = load_manifest_dataset(
        cfg.data.manifest_csv,
        file_col=cfg.data.file_col,
        subject_col=cfg.data.subject_col,
        label_col=cfg.data.label_col,
        device_col=cfg.data.device_col,
        session_col=cfg.data.session_col,
        feature_prefixes=cfg.data.feature_prefixes,
        device_filter=cfg.data.device_filter,
        session_filter=cfg.data.session_filter,
    )

    output_dir = Path(cfg.experiment.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_dim = cfg.model.input_dim or len(feature_cols)
    target_ids = cfg.experiment.target_subjects or sorted(subjects)

    rows = []
    for target_id in target_ids:
        target = subjects[target_id]
        sources = [s for sid, s in subjects.items() if sid != target_id]
        result = train_one_fold(
            sources,
            target,
            input_dim=input_dim,
            num_classes=cfg.model.num_classes,
            jsd_threshold=cfg.train.default_jsd_threshold,
            batch_size=cfg.train.batch_size,
            epochs=cfg.train.epochs,
            lr=cfg.train.learning_rate,
            weight_decay=cfg.train.weight_decay,
            kernel_mul=cfg.train.mmd_kernel_mul,
            kernel_num=cfg.train.mmd_kernel_num,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        fold_dir = output_dir / f"target_{target_id}"
        fold_dir.mkdir(exist_ok=True)
        torch.save(result["model"].state_dict(), fold_dir / "model.pt")
        pd.DataFrame(result["history"]).to_csv(fold_dir / "history.csv", index=False)
        np.savetxt(fold_dir / "confusion_matrix.csv", result["confusion_matrix"], delimiter=",", fmt="%d")
        (fold_dir / "selected_sources.json").write_text(
            json.dumps(result["selected_sources"], indent=2), encoding="utf-8"
        )
        (fold_dir / "jsd_scores.json").write_text(
            json.dumps(result["jsd_scores"], indent=2), encoding="utf-8"
        )
        report = classification_report(
            result["y_true"],
            result["y_pred"],
            target_names=label_encoder.classes_,
            zero_division=0,
        )
        (fold_dir / "classification_report.txt").write_text(report, encoding="utf-8")

        rows.append({
            "target_subject": target_id,
            "accuracy": result["accuracy"],
            "num_selected_sources": len(result["selected_sources"]),
            "selected_sources": ",".join(result["selected_sources"]),
        })
        print(f"target={target_id} accuracy={result['accuracy']:.4f}")

    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "loso_summary.csv", index=False)
    print(summary)
    print(f"mean_accuracy={summary['accuracy'].mean():.4f}")


if __name__ == "__main__":
    main()
