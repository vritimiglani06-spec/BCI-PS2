# src/analyze_results.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from config import LOG_DIR, RESULTS_DIR

def plot_confusion_matrices(label_type="valence"):
    print(f"\nGenerating aggregated confusion matrix for {label_type.upper()}...")
    all_true = []
    all_pred = []

    for fold in range(1, 24):
        pred_file = LOG_DIR / f"{label_type}_fold_{fold:02d}_predictions.csv"
        if pred_file.exists():
            df = pd.read_csv(pred_file)
            all_true.extend(df["true_label"].tolist())
            all_pred.extend(df["predicted_label"].tolist())

    cm = confusion_matrix(all_true, all_pred, normalize="true") * 100.0

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", cbar=True,
                xticklabels=["Low", "High"], yticklabels=["Low", "High"])
    plt.title(f"ASJDA Confusion Matrix: DREAMER {label_type.capitalize()} (%)")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    out_path = RESULTS_DIR / f"confusion_matrix_{label_type}.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")
    print(f"Class 0 Accuracy (Low): {cm[0, 0]:.2f}% | Class 1 Accuracy (High): {cm[1, 1]:.2f}%")

def plot_loss_trajectories(label_type="valence"):
    print(f"Generating loss curves for representative folds ({label_type.upper()})...")
    summary_file = RESULTS_DIR / f"dreamer_{label_type}_loso_summary.csv"
    if not summary_file.exists():
        print(f"Summary file {summary_file} not found.")
        return

    df_sum = pd.read_csv(summary_file)
    best_fold = int(df_sum.sort_values("test_accuracy", ascending=False).iloc[0]["target_id"])
    worst_fold = int(df_sum.sort_values("test_accuracy", ascending=True).iloc[0]["target_id"])
    median_idx = len(df_sum) // 2
    median_fold = int(df_sum.sort_values("test_accuracy").iloc[median_idx]["target_id"])

    rep_folds = [("Best", best_fold), ("Median", median_fold), ("Lowest", worst_fold)]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    for ax, (tag, fold_id) in zip(axes, rep_folds):
        loss_file = LOG_DIR / f"{label_type}_fold_{fold_id:02d}_loss.csv"
        df_loss = pd.read_csv(loss_file)
        
        ax.plot(df_loss["epoch"], df_loss["L_cls"], label=r"$\mathcal{L}_{cls}$", color="tab:blue")
        ax.plot(df_loss["epoch"], df_loss["L_mmd"], label=r"$\mathcal{L}_{mmd}$", color="tab:orange")
        ax.plot(df_loss["epoch"], df_loss["L_disc"], label=r"$\mathcal{L}_{disc}$", color="tab:green")
        ax.plot(df_loss["epoch"], df_loss["L_lsd"], label=r"$\mathcal{L}_{lsd}$", color="tab:red")
        ax.plot(df_loss["epoch"], df_loss["total_loss"], label=r"$\mathcal{L}_{total}$", color="black", linestyle="--")

        acc = df_sum.loc[df_sum["target_id"] == fold_id, "test_accuracy"].values[0]
        ax.set_title(f"{tag} Fold (S{fold_id:02d}) — Acc: {acc:.2f}%")
        ax.set_xlabel("Epoch")
        if tag == "Best":
            ax.set_ylabel("Loss Magnitude")
        ax.legend()
        ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    out_path = RESULTS_DIR / f"loss_curves_{label_type}.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")

def generate_report_tables():
    print("\n--- Compiling Benchmark & Replication Comparison ---")
    val_file = RESULTS_DIR / "dreamer_valence_loso_summary.csv"
    aro_file = RESULTS_DIR / "dreamer_arousal_loso_summary.csv"

    val_acc = f"{pd.read_csv(val_file)['test_accuracy'].mean():.2f}% +/- {pd.read_csv(val_file)['test_accuracy'].std():.2f}%" if val_file.exists() else "Missing"
    aro_acc = f"{pd.read_csv(aro_file)['test_accuracy'].mean():.2f}% +/- {pd.read_csv(aro_file)['test_accuracy'].std():.2f}%" if aro_file.exists() else "Missing"

    print(f"DREAMER Valence LOSO: {val_acc}")
    print(f"DREAMER Arousal LOSO: {aro_acc}\n")

if __name__ == "__main__":
    for dim in ["valence", "arousal"]:
        plot_confusion_matrices(dim)
        plot_loss_trajectories(dim)
    generate_report_tables()