# ASJDA Implementation for EEG Emotion Recognition

This is a standalone PyTorch implementation scaffold for the paper:

**Enhancing EEG-Based Cross-Subject Emotion Recognition via Adaptive Source Joint Domain Adaptation**

It now supports two dataset paths:

- **DEAP**, which is the better match because the ASJDA paper uses DEAP directly.
- The Zenodo low-cost/high-end EEG dataset, useful for a later cross-device extension.

## What Is Included

```text
asjda_implementation/
  asjda/
    config.py          configuration loader
    data.py            subject-wise feature dataset loader
    features.py        Differential Entropy feature extraction
    losses.py          JSD, MMD, DISC, LSD, dynamic weights
    model.py           ASJDA PyTorch architecture
    train.py           source selection and fold training
  configs/
    deap_arousal.yaml
    deap_valence.yaml
    zenodo_emotiv.yaml example experiment config
  docs/
    architecture.md    full architecture and implementation notes
  scripts/
    prepare_zenodo_features.py
    prepare_deap_dat.py
    download_kaggle_deap.py
    run_loso.py
  requirements.txt
```

## Best Dataset Choice

Use **DEAP first**. The paper evaluates ASJDA on DEAP using:

- 32 subjects
- 32 EEG channels
- 40 trials per subject
- 4-second EEG segments
- five Differential Entropy bands
- 160 features per segment: `32 channels * 5 bands`
- two binary tasks: valence and arousal

The Zenodo dataset is still useful, but it has two EEG devices and different
channel layouts. That makes it less direct for reproducing the paper.

## Expected Manifest Data Format

Create a manifest CSV:

```csv
subject_id,device,session,file
S01,emotiv,1,features/S01.csv
S02,emotiv,1,features/S02.csv
```

Each feature CSV should contain:

```csv
label,de_delta_ch00,de_delta_ch01,...,de_gamma_ch13
HVHA,0.12,0.33,...,0.20
LVLA,0.08,0.29,...,0.17
```

Labels for the Zenodo dataset should be:

- `HVHA`
- `HVLA`
- `LVHA`
- `LVLA`

## Install

```bash
cd outputs/asjda_implementation
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Download DEAP From Kaggle

Kaggle requires an account token. Put `kaggle.json` in:

```text
C:\Users\<you>\.kaggle\kaggle.json
```

Then run:

```bash
python scripts/download_kaggle_deap.py --unzip
```

You can also manually download from:

<https://www.kaggle.com/datasets/manh123df/deap-dataset>

## Prepare DEAP `.dat` Files

After download, find the folder containing `s01.dat` through `s32.dat`.

For valence:

```bash
python scripts/prepare_deap_dat.py ^
  --deap-dir path\to\deap-dataset\data_preprocessed_python ^
  --output-dir data ^
  --label-mode valence
```

For arousal:

```bash
python scripts/prepare_deap_dat.py ^
  --deap-dir path\to\deap-dataset\data_preprocessed_python ^
  --output-dir data ^
  --label-mode arousal
```

## Run DEAP Leave-One-Subject-Out Training

```bash
python scripts/run_loso.py --config configs/deap_valence.yaml
python scripts/run_loso.py --config configs/deap_arousal.yaml
```

Outputs are written to:

```text
runs/deap_valence_loso/
runs/deap_arousal_loso/
```

## Prepare Existing Feature CSVs

If the Zenodo archive already gives one feature CSV per subject:

```bash
python scripts/prepare_zenodo_features.py ^
  --input-dir path\to\zenodo\emotiv\features ^
  --output-dir data ^
  --device emotiv
```

If your features are not per-subject yet, split them first so every subject has
one CSV.

## Run Zenodo Leave-One-Subject-Out Training

```bash
python scripts/run_loso.py --config configs/zenodo_emotiv.yaml
```

Outputs are written to:

```text
runs/zenodo_emotiv_loso/
```

Each target subject gets:

- `model.pt`
- `history.csv`
- `confusion_matrix.csv`
- `classification_report.txt`
- `selected_sources.json`
- `jsd_scores.json`

## Recommended Experiment Order

1. DEAP valence.
2. DEAP arousal.
3. Zenodo Emotiv only.
4. Zenodo BrainVision only.
5. Cross-device after matched-channel feature harmonization.

ASJDA assumes every sample has the same feature dimension, so mixing BrainVision
and Emotiv directly will break or produce invalid comparisons.
