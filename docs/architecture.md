# ASJDA Architecture

This package implements the paper's Adaptive Source Joint Domain Adaptation
pipeline for cross-subject EEG emotion recognition.

## Data Flow

```text
Per-subject EEG/features
        |
        v
Differential Entropy feature matrix
        |
        v
Leave-one-subject-out split
        |
        +--> target subject: X_t, y_t used only for final evaluation
        |
        +--> source subjects: X_s, y_s
                 |
                 v
        JSD adaptive source selection
                 |
                 v
       selected source subjects S_1 ... S_K
                 |
                 v
       shared MLP feature extractor
                 |
                 v
       K domain-specific branches
                 |
                 v
       K source-specific classifiers
```

## Modules

- `asjda.data`: manifest-based subject loader and standardization.
- `asjda.features`: bandpass filtering and Differential Entropy extraction.
- `asjda.losses`: JSD, MMD, classifier discrepancy, LSD, dynamic weights.
- `asjda.model`: shared extractor, domain-specific branches, classifiers.
- `asjda.train`: source selection, one-fold training, target prediction.
- `scripts.run_loso`: full leave-one-subject-out experiment runner.
- `scripts.prepare_zenodo_features`: helper for normalizing existing feature CSVs.

## Network

The implemented network follows the architecture table in the paper:

```text
Shared feature extractor:
  Linear(input_dim, 128)
  LeakyReLU(0.01)
  Linear(128, 64)
  LeakyReLU(0.01)

For each selected source subject:
  Domain-specific extractor:
    Linear(64, 32)
    BatchNorm1d(32)
    LeakyReLU(0.01)

  Domain-specific classifier:
    Linear(32, num_classes)
```

For DEAP, use `num_classes = 2`:

- `low`: rating below or equal to the low side of the valence/arousal split
- `high`: rating above 5

For the Zenodo replacement dataset, use `num_classes = 4`:

- `HVHA`: high valence, high arousal
- `HVLA`: high valence, low arousal
- `LVHA`: low valence, high arousal
- `LVLA`: low valence, low arousal

## Losses

```text
L = L_cls + alpha * L_mmd + beta * L_disc + gamma * L_lsd
```

- `L_cls`: supervised source classification loss.
- `L_mmd`: global source-target domain alignment.
- `L_disc`: disagreement penalty across source-specific classifiers.
- `L_lsd`: category-level local subdomain alignment using target pseudo-labels.

Dynamic weights:

```text
alpha = 2 / (1 + exp(-10 * epoch / epochs)) - 1
beta = alpha / 100
gamma = alpha - 1
```

## Dataset Recommendation

DEAP is the better first dataset because it appears in the ASJDA paper's
experiments. It lets you keep the paper's feature shape and protocol:

```text
32 EEG channels * 5 DE bands = 160 input features
40 trials per subject -> 4-second segments -> 600 samples per subject
binary valence task and binary arousal task
```

## Important Zenodo Note

The Zenodo record contains BrainVision and Emotiv EPOC+ data. Run ASJDA on one
device at a time first, because the model expects source and target samples to
have the same feature dimension. Cross-device training needs channel matching or
another harmonization layer before ASJDA.
