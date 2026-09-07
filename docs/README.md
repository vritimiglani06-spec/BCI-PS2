# ASJDA DEAP Replication — Documentation



## Overview



This document contains the detailed documentation for the DEAP-only implementation of **Adaptive Source Joint Domain Adaptation (ASJDA)** for cross-subject EEG emotion recognition.



The implementation is based on the paper:



**Enhancing EEG-Based Cross-Subject Emotion Recognition via Adaptive Source Joint Domain Adaptation (ASJDA)**



The current repository focuses only on the **DEAP dataset**.



---



## 1. DEAP Dataset



DEAP is used for cross-subject EEG emotion recognition.



### Dataset characteristics



| Property | Value |

|---|---|

| Subjects | 32 |

| EEG channels used | 32 |

| Trials per subject | 40 |

| Sampling rate | 128 Hz |

| Frequency range | 4–45 Hz |

| Frequency bands | 5 |

| Feature type | Differential Entropy (DE) |

| Feature dimension | 160 |

| Tasks | Valence, Arousal |

| Evaluation | Leave-One-Subject-Out |



The continuous DEAP ratings are converted into binary classes using a threshold of 5:



\- Score > 5 → High

\- Score < 5 → Low

\- Score = 5 → Removed



---



## 2. Replication Pipeline



The overall workflow is:



```text

DEAP .dat files

      |

      v

Subject-wise preprocessing

      |

      v

EEG channel selection

      |

      v

Differential Entropy extraction

      |

      v

160-dimensional feature vectors

      |

      v

Subject-wise feature CSV files

      |

      v

Manifest CSV

      |

      v

Leave-One-Subject-Out split

      |

      v

Source / target domains

      |

      v

Adaptive source selection

      |

      v

Shared feature extraction

      |

      v

Subdomain-level adaptation

      |

      v

Category-level adaptation

      |

      v

Source-specific classification

      |

      v

Target prediction

      |

      v

LOSO evaluation

```


---

## 3. Data Preparation

The original DEAP `.dat` files are not stored in the Git repository because of their size.

The local DEAP directory should contain:

```

data_preprocessed_python/

â”œâ”€â”€ s01.dat

â”œâ”€â”€ s02.dat

â”œâ”€â”€ ...

â””â”€â”€ s32.dat


```


### Valence

Run:

```
python scripts/prepare_deap_dat.py

  --deap-dir D:\\DEAP\\full_extract\\deap-dataset\\data_preprocessed_python `

  --output-dir D:\\DEAP\\asjda_data `

  --label-mode valence

```


### Arousal

Run:

```
python scripts/prepare_deap_dat.py

  --deap-dir D:\\DEAP\\full_extract\\deap-dataset\\data_preprocessed_python `

  --output-dir D:\\DEAP\\asjda_data `

  --label-mode arousal

```


The preparation process creates subject-level feature files and a manifest.

Example:

```

asjda_data/

â”œâ”€â”€ manifest_deap_valence.csv

â”œâ”€â”€ manifest_deap_arousal.csv

â”œâ”€â”€ features_deap_valence/

â”‚   â”œâ”€â”€ s01.csv

â”‚   â”œâ”€â”€ s02.csv

â”‚   â””â”€â”€ ...

â””â”€â”€ features_deap_arousal/

    â”œâ”€â”€ s01.csv

    â”œâ”€â”€ s02.csv

    â””â”€â”€ ...


```


---

## 4. Manifest

The manifest stores the subject identifier, dataset information, label mode, and feature-file path.

Example:

```
subject_id,device,session,label_mode,file

s01,deap_biosemi,1,valence,features_deap_valence\\s01.csv

s02,deap_biosemi,1,valence,features_deap_valence\\s02.csv

...

s32,deap_biosemi,1,valence,features_deap_valence\\s32.csv

```


Relative paths are resolved relative to the directory containing the manifest.

---

## 5. Model Architecture

The current DEAP implementation uses a shared feature extractor followed by source-specific feature extractors and classifiers.

```

Input

160 DE features

      |

      v

Shared Feature Extractor

      |

      v

128 hidden units

      |

      v

64-dimensional representation

      |

      +------------------+

      |                  |

      v                  v

Source Feature       Source Feature

Extractor 1          Extractor 2

      |                  |

     ...                ...

      |                  |

      v                  v

32-dimensional       32-dimensional

representation       representation

      |                  |

      v                  v

Classifier 1         Classifier 2

      |                  |

      +--------+---------+

               |

               v

        Target prediction


```


The shared extractor learns a representation common across subjects.

The source-specific components allow the model to retain domain-specific information for individual source subjects.

---

## 6. Adaptive Source Selection

One of the main ideas of ASJDA is that not every source subject should necessarily be used for every target subject.

For a given target subject:

```

All other subjects

        |

        v

Calculate source-target domain similarity

        |

        v

Jensen-Shannon Divergence

        |

        v

Select sufficiently similar sources

        |

        v

Use selected sources for adaptation


```


The goal is to reduce negative transfer from source subjects whose EEG distributions are substantially different from the target.

---

## 7. Subdomain-Level Adaptation

The implementation performs domain alignment between source and target feature representations.

The subdomain-level component uses an MMD-based loss.

Conceptually:

```

Source representation

        |

        |\\

        | \\

        |  --> MMD loss

        | /

        |/

Target representation


```


The objective is to reduce the distribution difference between source and target representations.

---

## 8. Category-Level Adaptation

The model also uses prediction information for category-level adaptation.

Source-specific classifiers produce predictions, and confident predictions can be used to obtain pseudo-label information.

The general process is:

```

Source / target features

        |

        v

Source-specific classifiers

        |

        v

Predicted class probabilities

        |

        v

Confidence filtering

        |

        v

Pseudo-label information

        |

        v

Category-level alignment


```


This encourages features belonging to corresponding emotion categories to become more aligned across domains.

---

## 9. Training Objective

The overall objective combines classification and domain-adaptation terms:

```

Total Loss

    =

Classification Loss

    +

Subdomain Adaptation Loss

    +

Category Adaptation Loss


```


The implementation of the individual loss components is contained in:

```

asjda/losses.py


```


The training procedure is implemented in:

```

asjda/train.py


```


---

## 10. Leave-One-Subject-Out Evaluation

The evaluation uses **Leave-One-Subject-Out (LOSO)** cross-subject evaluation.

For each fold:

```

One subject → Target



All remaining subjects → Sources


```


For DEAP this produces:

```

32 LOSO folds


```


For example:

```

Fold 1:

Target = S01

Sources = S02–S32



Fold 2:

Target = S02

Sources = S01, S03–S32



...



Fold 32:

Target = S32

Sources = S01–S31


```


The target subject is not used as a labeled source during training.

---

## 11. Configuration Files

The experiment configurations are stored in:

```

configs/

â”œâ”€â”€ deap_valence.yaml

â””â”€â”€ deap_arousal.yaml


```


The configuration files define the dataset, label mode, training parameters, and experiment settings.

---

## 12. Running the Experiments

The LOSO experiment is launched using:

```
python scripts/run_loso.py

```


The script uses the selected configuration to prepare the source and target domains and perform the LOSO experiment.

---

## 13. Experiment Outputs

Experiment outputs are stored in:

```

runs/


```


The `runs/` directory is excluded from Git because experiment outputs can become large.

Typical outputs include:

* fold-level predictions

* target-subject accuracy

* classification metrics

* confusion matrices

* experiment summaries

---

## 14. Result Visualization

The repository contains:

```

scripts/plot_loso_results.py


```


This script is used to visualize LOSO experiment results after the experiments have been completed.

---

## 15. Current Replication Status

The DEAP replication currently includes:

* [x] DEAP dataset preparation

* [x] Subject-wise preprocessing

* [x] Differential Entropy feature preparation

* [x] 160-dimensional DE features

* [x] Manifest generation

* [x] DEAP data loading

* [x] LOSO evaluation

* [x] Shared feature extraction

* [x] Source-specific feature extraction

* [x] Source-specific classification

* [x] Domain-adaptation components

* [x] Valence experiment

* [x] Arousal experiment

* [ ] Exact numerical reproduction of every result reported in the paper

* [ ] Full reproduction of all baseline methods

Completion of the implementation does not necessarily mean that every reported value from the original paper will be reproduced exactly.

---

## 16. Reproducibility

A complete local reproduction should follow this sequence:

```

1. Obtain DEAP

        |

        v

2. Prepare subject data

        |

        v

3. Generate DE features

        |

        v

4. Generate manifest

        |

        v

5. Select valence or arousal

        |

        v

6. Run LOSO

        |

        v

7. Save experiment results

        |

        v

8. Generate summary statistics

        |

        v

9. Plot results


```


Random seeds should be fixed when deterministic or comparable experiments are required.

---

## 17. Repository Structure

```

ASJDA-EEG-Replication/

â”‚

â”œâ”€â”€ README.md

â”œâ”€â”€ requirements.txt

â”‚

â”œâ”€â”€ asjda/

â”‚   â”œâ”€â”€ __init__.py

â”‚   â”œâ”€â”€ config.py

â”‚   â”œâ”€â”€ data.py

â”‚   â”œâ”€â”€ features.py

â”‚   â”œâ”€â”€ losses.py

â”‚   â”œâ”€â”€ model.py

â”‚   â””â”€â”€ train.py

â”‚

â”œâ”€â”€ configs/

â”‚   â”œâ”€â”€ deap_arousal.yaml

â”‚   â””â”€â”€ deap_valence.yaml

â”‚

â”œâ”€â”€ scripts/

â”‚   â”œâ”€â”€ prepare_deap_dat.py

â”‚   â”œâ”€â”€ download_kaggle_deap.py

â”‚   â”œâ”€â”€ run_loso.py

â”‚   â””â”€â”€ plot_loso_results.py

â”‚

â””â”€â”€ docs/

    â”œâ”€â”€ README.md

    â”œâ”€â”€ architecture.md

    â””â”€â”€ ASJDA_DEAP_Replication_Report_Detailed.md


```


---

## 18. Git and Dataset Management

Large dataset files are intentionally excluded from version control.

The repository `.gitignore` excludes:

```

*.dat

*.zip

*.rar

*.mat

runs/


```


This keeps the repository lightweight.

The dataset must therefore be obtained and prepared separately before running the experiments.

---

## 19. Important Implementation Note

The current implementation is a cleaned DEAP-focused replication.

The repository should be interpreted as an implementation and experimental reproduction of the ASJDA approach rather than a claim that every implementation detail and numerical result is identical to the original authors' environment.

Differences can arise from:

* preprocessing

* feature extraction

* normalization

* source selection

* random initialization

* training settings

* optimization

* hardware

* numerical precision

* implementation details

Experimental results should therefore always be reported together with the configuration used to obtain them.

---

## 20. Reference

**Paper:**

*Enhancing EEG-Based Cross-Subject Emotion Recognition via Adaptive Source Joint Domain Adaptation (ASJDA)*

**Journal:**

IEEE Transactions on Affective Computing

The implementation in this repository is intended for academic replication, experimentation, and further development.
