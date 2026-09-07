# ASJDA Architecture

This package implements the paper's Adaptive Source Joint Domain Adaptation
pipeline for cross-subject EEG emotion recognition, evaluated on the DEAP
dataset.

## Data Flow

```text
Per-subject EEG
      |
      v
Differential Entropy feature matrix
      |
      v
Leave-One-Subject-Out split
      |
      +--> target subject: X_t, y_t
      |        |
      |        +--> used only for final evaluation
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
       shared feature extractor
               |
               v
       K domain-specific branches
               |
               v
       K source-specific classifiers
               |
               v
       target prediction