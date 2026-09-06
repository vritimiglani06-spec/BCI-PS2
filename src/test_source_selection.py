# src/test_source_selection.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from config import DATA_PROCESSED, RESULTS_DIR
from source_selector import SourceSelector

def run_test():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading extracted DE features and labels...")
    features = {s: np.load(DATA_PROCESSED / f"DREAMER_DE_S{s:02d}.npy") for s in range(1, 24)}
    val_labels = {s: np.load(DATA_PROCESSED / f"DREAMER_labels_valence_S{s:02d}.npy") for s in range(1, 24)}
    aro_labels = {s: np.load(DATA_PROCESSED / f"DREAMER_labels_arousal_S{s:02d}.npy") for s in range(1, 24)}

    selector = SourceSelector(n_components=10, n_bins=20)
    print("Fitting PCA and calculating 23x23 pairwise JSD matrix...")
    jsd_matrix = selector.fit(features)

    # Summary statistics
    upper_tri = jsd_matrix[np.triu_indices(23, k=1)]
    print(f"\nJSD Distribution Statistics across all pairs (N={len(upper_tri)}):")
    print(f"Min: {np.min(upper_tri):.5f} | 25th %: {np.percentile(upper_tri, 25):.5f} | Median: {np.median(upper_tri):.5f}")
    print(f"75th %: {np.percentile(upper_tri, 75):.5f} | Max: {np.max(upper_tri):.5f}")

    # Plot JSD heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(jsd_matrix, cmap="YlGnBu", xticklabels=range(1, 24), yticklabels=range(1, 24))
    plt.title("Pairwise Jensen-Shannon Divergence (DREAMER)")
    plt.xlabel("Subject ID")
    plt.ylabel("Subject ID")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "pairwise_jsd_matrix.png", dpi=300)
    print(f"Heatmap saved to {RESULTS_DIR / 'pairwise_jsd_matrix.png'}")

    # Generate candidate threshold range from empirical percentiles
    candidates = np.linspace(np.percentile(upper_tri, 20), np.percentile(upper_tri, 85), 10)
    print(f"\nCandidate threshold range: {[round(float(c), 4) for c in candidates]}")

    print("\n--- Testing Source Selection per Target Subject (VALENCE) ---")
    val_counts = []
    for target in range(1, 24):
        best_t = selector.calibrate_threshold(target, features, val_labels, candidates)
        selected = selector.select_sources(target, best_t)
        val_counts.append(len(selected))
        print(f"Target S{target:02d}: Optimal Threshold={best_t:.4f} | Selected K={len(selected)} sources")

    print(f"\nMean Sources Selected (Valence): {np.mean(val_counts):.1f} +/- {np.std(val_counts):.1f} (out of 22)")

if __name__ == "__main__":
    run_test()