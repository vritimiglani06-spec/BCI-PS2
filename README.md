# Independent Replication: ASJDA on the DREAMER Dataset

Clean-room independent replication and methodological audit of:
> **Adaptive Source Joint Domain Adaptation for EEG Cross-subject Emotion Recognition** (*IEEE Transactions on Affective Computing*).

This repository provides an automated, hardware-optimized pipeline executing the ASJDA architecture on the **DREAMER dataset** (23 subjects, 14-channel Emotiv EPOC EEG). The engine features GPU-resident tensor slicing and asynchronous loss evaluation optimized for consumer GPUs (validated on an NVIDIA RTX 3050 Laptop GPU 6GB).

---

## Key Replication Findings

* **Prior Matching Over Emotion Decoding:** Full ASJDA achieves **58.43% ± 8.83%** on Valence and **51.76% ± 9.28%** on Arousal. These track the raw dataset background priors (**58.90%** and **52.05%**), rather than true affective features.
* **Majority Prior Collapse:** Aggregated confusion matrices yield a balanced accuracy of **50.77% (Valence)** and **50.85% (Arousal)**. The network suffered from confirmation bias in the Local Sub-domain Discrepancy (LSD) pseudo-labeling loop, collapsing into predicting the majority class (predicting Low Valence 93.78% of the time).
* **Ablation Invariance:** A non-parametric Friedman test across all four Valence experimental variants yields $Q = 0.9605$ ($p = 0.8108$). Removing MMD, LSD, or Adaptive Source Selection produced zero statistically significant performance degradation.
* **Codebase Discrepancies:** Inspection of the authors' official repository uncovered that the published paper deviates significantly from the operational code: linear MMD is used instead of multi-scale Gaussian RBF, consensus discrepancy is computed across 32-D latent feature coordinates rather than class probabilities, and the Adam optimizer is re-instantiated inside the training loop.

---

## Directory Structure

```text
asjda_replication/
├── data/
│   ├── raw/
│   │   └── DREAMER.mat
│   └── processed/
│       ├── DREAMER_DE_S01.npy ... S23.npy
│       ├── DREAMER_labels_valence_S01.npy ... S23.npy
│       └── DREAMER_labels_arousal_S01.npy ... S23.npy
├── checkpoints/
│   ├── valence_fold_*.pt
│   └── arousal_fold_*.pt
├── logs/
│   ├── valence_fold_*_loss.csv
│   ├── valence_fold_*_predictions.csv
│   ├── arousal_fold_*_loss.csv
│   └── arousal_fold_*_predictions.csv
├── results/
│   ├── confusion_matrix_valence.png
│   ├── confusion_matrix_arousal.png
│   ├── loss_curves_valence.png
│   ├── loss_curves_arousal.png
│   ├── jsd_distribution_histogram.png
│   ├── sources_selected_per_target.png
│   ├── pairwise_jsd_matrix.png
│   ├── dreamer_valence_loso_summary.csv
│   ├── dreamer_arousal_loso_summary.csv
│   ├── dreamer_valence_ablation_no_lsd.csv
│   ├── dreamer_valence_ablation_no_selection.csv
│   ├── dreamer_valence_ablation_no_mmd.csv
│   └── MASTER_RESULTS_TABLE.md
├── src/
│   ├── config.py
│   ├── preprocess_dreamer.py
│   ├── source_selector.py
│   ├── models.py
│   ├── train_loso.py
│   ├── run_ablations.py
│   ├── analyze_results.py
│   ├── plot_source_selection.py
│   └── compile_tables.py
├── README.md
└── REPLICATION_REPORT.md