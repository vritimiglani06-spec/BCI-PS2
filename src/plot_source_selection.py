# src/plot_source_selection.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config import DATA_PROCESSED, RESULTS_DIR
from source_selector import SourceSelector

def generate_source_selection_plots():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading features and summary files for visualization...")
    
    features_cpu = {s: np.load(DATA_PROCESSED / f"DREAMER_DE_S{s:02d}.npy") for s in range(1, 24)}
    val_summary = pd.read_csv(RESULTS_DIR / "dreamer_valence_loso_summary.csv")
    aro_summary = pd.read_csv(RESULTS_DIR / "dreamer_arousal_loso_summary.csv")

    selector = SourceSelector(n_components=10, n_bins=20)
    jsd_matrix = selector.fit(features_cpu)
    upper_tri = jsd_matrix[np.triu_indices(23, k=1)]

    # 1. JSD Distribution Histogram
    plt.figure(figsize=(8, 5))
    sns.histplot(upper_tri, bins=25, kde=True, color="steelblue", edgecolor="black", alpha=0.7)
    plt.axvline(np.mean(val_summary["threshold"]), color="red", linestyle="--", linewidth=2, 
                label=f"Mean Valence Threshold ({val_summary['threshold'].mean():.4f})")
    plt.axvline(np.mean(aro_summary["threshold"]), color="green", linestyle=":", linewidth=2, 
                label=f"Mean Arousal Threshold ({aro_summary['threshold'].mean():.4f})")
    plt.title("Distribution of Pairwise Jensen-Shannon Divergence (DREAMER)")
    plt.xlabel("JSD Value")
    plt.ylabel("Pair Count")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    hist_path = RESULTS_DIR / "jsd_distribution_histogram.png"
    plt.savefig(hist_path, dpi=300)
    plt.close()
    print(f"Saved: {hist_path}")

    # 2. Per-Target Source Count (K) Comparison Bar Chart
    plt.figure(figsize=(12, 5))
    x = np.arange(1, 24)
    width = 0.4

    plt.bar(x - width/2, val_summary["n_sources_selected"], width, label="Valence Selected K", color="#4C72B0")
    plt.bar(x + width/2, aro_summary["n_sources_selected"], width, label="Arousal Selected K", color="#55A868")
    plt.axhline(22, color="gray", linestyle="--", alpha=0.7, label="Max Available Sources (K=22)")

    plt.title("Number of Selected Sources (K) per Target Subject")
    plt.xlabel("Target Subject ID")
    plt.ylabel("Number of Sources Selected (K)")
    plt.xticks(x)
    plt.legend()
    plt.grid(True, axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()
    bar_path = RESULTS_DIR / "sources_selected_per_target.png"
    plt.savefig(bar_path, dpi=300)
    plt.close()
    print(f"Saved: {bar_path}")

if __name__ == "__main__":
    generate_source_selection_plots()