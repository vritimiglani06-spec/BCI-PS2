# ASJDA: EEG-Based Cross-Subject Emotion Recognition

A PyTorch implementation and replication study of the paper:

> **Enhancing EEG-Based Cross-Subject Emotion Recognition via Adaptive Source Joint Domain Adaptation (ASJDA)**

This repository focuses on reproducing and evaluating the ASJDA framework on the **DEAP dataset** for cross-subject EEG emotion recognition.

---

## Overview

Electroencephalography (EEG)-based emotion recognition is strongly affected by differences between subjects. EEG signals collected from different individuals can have substantially different distributions, making it difficult for a model trained on one group of subjects to generalize to an unseen subject.

The **Adaptive Source Joint Domain Adaptation (ASJDA)** framework addresses this cross-subject domain shift by combining:

- Adaptive source-domain selection using **Jensen-Shannon Divergence (JSD)**
- Shared feature extraction
- Source-specific domain feature extraction
- Global domain alignment using **Maximum Mean Discrepancy (MMD)**
- Source-specific classifier discrepancy minimization
- Local subdomain alignment using **pseudo-labels**

The implementation in this repository evaluates these ideas using a **Leave-One-Subject-Out (LOSO)** cross-subject protocol on DEAP.

---

## Dataset: DEAP

The experiments use the **DEAP dataset**, a multimodal dataset for the analysis of human affective states.

For this project, EEG data is processed into Differential Entropy (DE) features.

### Dataset characteristics

| Property | Value |
|---|---:|
| Subjects | 32 |
| EEG channels | 32 |
| Trials per subject | 40 |
| Frequency bands | 5 |
| DE features | 160 |
| Tasks | Valence, Arousal |
| Classification | Binary |
| Evaluation | Leave-One-Subject-Out |

The feature representation is:

```text
32 EEG channels × 5 frequency bands
              ↓
        160 DE features
```

Two binary classification tasks are considered:

- **Valence**
- **Arousal**

The labels are divided using the rating threshold:

```text
low   : rating <= 5
high  : rating > 5
```

---

## ASJDA Pipeline

The overall cross-subject pipeline is:

```text
                    DEAP EEG
                       │
                       ▼
             Differential Entropy
                  Extraction
                       │
                       ▼
              160-D Feature Vector
                       │
                       ▼
          Leave-One-Subject-Out Split
                       │
             ┌─────────┴─────────┐
             │                   │
             ▼                   ▼
      Source Subjects        Target Subject
             │                   │
             ▼                   │
      JSD Source Selection        │
             │                   │
             ▼                   │
      Selected Sources            │
             │                   │
             └─────────┬─────────┘
                       ▼
             Shared Feature Extractor
                       │
                       ▼
          Source-Specific Feature Branches
                       │
                       ▼
            Source-Specific Classifiers
                       │
                       ▼
               Target Prediction
```

The target subject is held out during training and is used for final evaluation.

---

## Network Architecture

The model contains a shared feature extractor followed by source-specific domain branches and classifiers.

### Shared feature extractor

```text
Input: 160 DE features
        │
        ▼
Linear(160 → 128)
        │
        ▼
LeakyReLU(0.01)
        │
        ▼
Linear(128 → 64)
        │
        ▼
LeakyReLU(0.01)
```

### Domain-specific feature extractor

For every selected source subject:

```text
64-dimensional shared representation
        │
        ▼
Linear(64 → 32)
        │
        ▼
BatchNorm1d(32)
        │
        ▼
LeakyReLU(0.01)
```

### Source-specific classifier

```text
32-dimensional domain representation
        │
        ▼
Linear(32 → 2)
        │
        ▼
Class prediction
```

---

## Domain Adaptation Objective

The training objective combines supervised classification with domain adaptation losses:

```text
L = L_cls + α L_mmd + β L_disc + γ L_lsd
```

where:

- `L_cls` — supervised source classification loss
- `L_mmd` — global source-target domain alignment
- `L_disc` — discrepancy between source-specific classifiers
- `L_lsd` — local subdomain alignment using pseudo-labels

The dynamic loss weights are defined as:

```text
α = 2 / (1 + exp(-10 × epoch / epochs)) - 1

β = α / 100

γ = α - 1
```

---

## Project Structure

```text
asjda_implementation/
│
├── README.md
├── requirements.txt
│
├── asjda/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── features.py
│   ├── losses.py
│   ├── model.py
│   └── train.py
│
├── configs/
│   ├── deap_arousal.yaml
│   └── deap_valence.yaml
│
├── docs/
│   ├── README.md
│   ├── architecture.md
│   └── ASJDA_DEAP_Replication_Report_Detailed.md
│
└── scripts/
    ├── download_kaggle_deap.py
    ├── prepare_deap_dat.py
    ├── plot_loso_results.py
    └── run_loso.py
```

---

## Installation

Clone the repository and move into the project directory:

```bash
git clone <repository-url>
cd asjda_implementation
```

Create a Python virtual environment:

### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## DEAP Data Preparation

The raw DEAP dataset is **not included in this repository**.

After obtaining the DEAP preprocessed data, locate the directory containing:

```text
s01.dat
s02.dat
s03.dat
...
s32.dat
```

The `.dat` files contain the preprocessed EEG recordings and emotion ratings.

### Generate Valence Features

```powershell
python scripts/prepare_deap_dat.py `
  --deap-dir path\to\data_preprocessed_python `
  --output-dir data `
  --label-mode valence
```

### Generate Arousal Features

```powershell
python scripts/prepare_deap_dat.py `
  --deap-dir path\to\data_preprocessed_python `
  --output-dir data `
  --label-mode arousal
```

The preparation step generates subject-level feature files and a manifest used by the ASJDA training pipeline.

---

## Running the Experiments

### Valence

Run the DEAP valence LOSO experiment:

```bash
python scripts/run_loso.py --config configs/deap_valence.yaml
```

### Arousal

Run the DEAP arousal LOSO experiment:

```bash
python scripts/run_loso.py --config configs/deap_arousal.yaml
```

Each experiment performs a 32-fold LOSO evaluation:

```text
Fold 1   → Subject 1 as target
Fold 2   → Subject 2 as target
...
Fold 32  → Subject 32 as target
```

For every fold:

```text
31 subjects → source domains
1 subject   → target domain
```

---

## Output Files

Experiment outputs are stored under:

```text
runs/
```

Typical output structure:

```text
runs/
├── deap_valence_loso/
│   ├── loso_summary.csv
│   ├── ASJDA_DEAP_LOSO_accuracy.png
│   ├── s01/
│   ├── s02/
│   ├── ...
│   └── s32/
│
└── deap_arousal_loso/
    ├── loso_summary.csv
    ├── ASJDA_DEAP_LOSO_accuracy.png
    ├── s01/
    ├── s02/
    ├── ...
    └── s32/
```

Individual folds may contain:

```text
model.pt
history.csv
confusion_matrix.csv
classification_report.txt
selected_sources.json
jsd_scores.json
```

The `runs/` directory and raw datasets are excluded from version control.

---

## Result Visualization

After running an experiment, the LOSO results can be visualized using:

```bash
python scripts/plot_loso_results.py
```

The resulting plots can be used to examine subject-wise accuracy and overall LOSO performance.

---

## Reproducibility Workflow

The recommended order for reproducing the experiments is:

```text
1. Obtain DEAP
        ↓
2. Prepare DEAP features
        ↓
3. Generate valence features
        ↓
4. Run valence LOSO
        ↓
5. Inspect results
        ↓
6. Generate arousal features
        ↓
7. Run arousal LOSO
        ↓
8. Inspect results
        ↓
9. Compare with the ASJDA paper
        ↓
10. Perform additional implementation analysis / ablations
```

---

## Current Replication Focus

This project is primarily focused on:

- DEAP preprocessing
- Differential Entropy feature extraction
- Cross-subject emotion recognition
- Adaptive source selection
- Domain adaptation
- Leave-One-Subject-Out evaluation
- Comparison with the reported ASJDA results
- Analysis of implementation details and methodological differences

The detailed replication findings are documented separately in:

```text
docs/ASJDA_DEAP_Replication_Report_Detailed.md
```

---

## Documentation

Additional documentation is available in the `docs/` directory.

### Architecture

See:

```text
docs/architecture.md
```

for the model architecture, data flow, and loss formulation.

### Detailed Replication Report

See:

```text
docs/ASJDA_DEAP_Replication_Report_Detailed.md
```

for the detailed DEAP replication report, experimental observations, results, and implementation analysis.

### Detailed Documentation

See:

```text
docs/README.md
```

for the extended project documentation and experiment instructions.

---

## Important Notes

### Raw data

The DEAP dataset is not distributed with this repository.

Users must obtain the dataset separately and prepare the required subject-level features before running the experiments.

### Experiment outputs

Large experiment outputs, trained model files, and generated runs are excluded from Git.

### Cross-subject evaluation

The project uses a strict LOSO structure in which the target subject is held out from the source training domains and evaluated after training.

---

## Reference

This implementation is based on:

> **Enhancing EEG-Based Cross-Subject Emotion Recognition via Adaptive Source Joint Domain Adaptation (ASJDA)**

The goal of this repository is to reproduce the DEAP-based ASJDA experiments and document the implementation and experimental findings.

---

## License

This repository is intended for academic and research purposes.

Please refer to the original dataset and paper for their respective licensing and usage conditions.