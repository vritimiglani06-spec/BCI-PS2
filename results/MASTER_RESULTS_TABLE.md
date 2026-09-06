# Master Replication Results Table

| Experiment / Configuration | Metric | Accuracy (Mean ± Std) | Baseline Delta |
| :--- | :--- | :--- | :--- |
| **DREAMER Valence (Full ASJDA)** | 23-Fold LOSO | **58.43% ± 8.83%** | Baseline |
| **DREAMER Arousal (Full ASJDA)** | 23-Fold LOSO | **51.76% ± 9.28%** | Baseline |
| **Valence w/o LSD (`no_lsd`)** | Ablation | **59.02% ± 12.02%** | +0.59% |
| **Valence w/o Selection (`no_selection`)** | Ablation | **58.73% ± 8.46%** | +0.30% |
| **Valence w/o MMD (`no_mmd`)** | Ablation | **58.70% ± 7.94%** | +0.27% |
| **DREAMER Valence Trivial Prior** | Majority Baseline | **58.90%** | +0.47% |
| **DREAMER Arousal Trivial Prior** | Majority Baseline | **52.05%** | +0.29% |
