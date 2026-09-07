````markdown

\# ASJDA DEAP Replication â€” Documentation



\## Overview



This document contains the detailed documentation for the DEAP-only implementation of \*\*Adaptive Source Joint Domain Adaptation (ASJDA)\*\* for cross-subject EEG emotion recognition.



The implementation is based on the paper:



\*\*Enhancing EEG-Based Cross-Subject Emotion Recognition via Adaptive Source Joint Domain Adaptation (ASJDA)\*\*



The current repository focuses only on the **DEAP dataset**.



\---



\## 1. DEAP Dataset



DEAP is used for cross-subject EEG emotion recognition.



\### Dataset characteristics



| Property | Value |

|---|---|

| Subjects | 32 |

| EEG channels used | 32 |

| Trials per subject | 40 |

| Sampling rate | 128 Hz |

| Frequency range | 4â€“45 Hz |

| Frequency bands | 5 |

| Feature type | Differential Entropy (DE) |

| Feature dimension | 160 |

| Tasks | Valence, Arousal |

| Evaluation | Leave-One-Subject-Out |



The continuous DEAP ratings are converted into binary classes using a threshold of 5:



\- Score > 5 â†’ High

\- Score < 5 â†’ Low

\- Score = 5 â†’ Removed



\---



\## 2. Replication Pipeline



The overall workflow is:



```text

DEAP .dat files

&#x20;     |

&#x20;     v

Subject-wise preprocessing

&#x20;     |

&#x20;     v

EEG channel selection

&#x20;     |

&#x20;     v

Differential Entropy extraction

&#x20;     |

&#x20;     v

160-dimensional feature vectors

&#x20;     |

&#x20;     v

Subject-wise feature CSV files

&#x20;     |

&#x20;     v

Manifest CSV

&#x20;     |

&#x20;     v

Leave-One-Subject-Out split

&#x20;     |

&#x20;     v

Source / target domains

&#x20;     |

&#x20;     v

Adaptive source selection

&#x20;     |

&#x20;     v

Shared feature extraction

&#x20;     |

&#x20;     v

Subdomain-level adaptation

&#x20;     |

&#x20;     v

Category-level adaptation

&#x20;     |

&#x20;     v

Source-specific classification

&#x20;     |

&#x20;     v

Target prediction

&#x20;     |

&#x20;     v

LOSO evaluation

````



\---



\## 3. Data Preparation



The original DEAP `.dat` files are not stored in the Git repository because of their size.



The local DEAP directory should contain:



```text

data\_preprocessed\_python/

â”œâ”€â”€ s01.dat

â”œâ”€â”€ s02.dat

â”œâ”€â”€ ...

â””â”€â”€ s32.dat

```



\### Valence



Run:



```powershell

python scripts/prepare\_deap\_dat.py `

&#x20; --deap-dir D:\\DEAP\\full\_extract\\deap-dataset\\data\_preprocessed\_python `

&#x20; --output-dir D:\\DEAP\\asjda\_data `

&#x20; --label-mode valence

```



\### Arousal



Run:



```powershell

python scripts/prepare\_deap\_dat.py `

&#x20; --deap-dir D:\\DEAP\\full\_extract\\deap-dataset\\data\_preprocessed\_python `

&#x20; --output-dir D:\\DEAP\\asjda\_data `

&#x20; --label-mode arousal

```



The preparation process creates subject-level feature files and a manifest.



Example:



```text

asjda\_data/

â”œâ”€â”€ manifest\_deap\_valence.csv

â”œâ”€â”€ manifest\_deap\_arousal.csv

â”œâ”€â”€ features\_deap\_valence/

â”‚   â”œâ”€â”€ s01.csv

â”‚   â”œâ”€â”€ s02.csv

â”‚   â””â”€â”€ ...

â””â”€â”€ features\_deap\_arousal/

&#x20;   â”œâ”€â”€ s01.csv

&#x20;   â”œâ”€â”€ s02.csv

&#x20;   â””â”€â”€ ...

```



\---



\## 4. Manifest



The manifest stores the subject identifier, dataset information, label mode, and feature-file path.



Example:



```csv

subject\_id,device,session,label\_mode,file

s01,deap\_biosemi,1,valence,features\_deap\_valence\\s01.csv

s02,deap\_biosemi,1,valence,features\_deap\_valence\\s02.csv

...

s32,deap\_biosemi,1,valence,features\_deap\_valence\\s32.csv

```



Relative paths are resolved relative to the directory containing the manifest.



\---



\## 5. Model Architecture



The current DEAP implementation uses a shared feature extractor followed by source-specific feature extractors and classifiers.



```text

Input

160 DE features

&#x20;     |

&#x20;     v

Shared Feature Extractor

&#x20;     |

&#x20;     v

128 hidden units

&#x20;     |

&#x20;     v

64-dimensional representation

&#x20;     |

&#x20;     +------------------+

&#x20;     |                  |

&#x20;     v                  v

Source Feature       Source Feature

Extractor 1          Extractor 2

&#x20;     |                  |

&#x20;    ...                ...

&#x20;     |                  |

&#x20;     v                  v

32-dimensional       32-dimensional

representation       representation

&#x20;     |                  |

&#x20;     v                  v

Classifier 1         Classifier 2

&#x20;     |                  |

&#x20;     +--------+---------+

&#x20;              |

&#x20;              v

&#x20;       Target prediction

```



The shared extractor learns a representation common across subjects.



The source-specific components allow the model to retain domain-specific information for individual source subjects.



\---



\## 6. Adaptive Source Selection



One of the main ideas of ASJDA is that not every source subject should necessarily be used for every target subject.



For a given target subject:



```text

All other subjects

&#x20;       |

&#x20;       v

Calculate source-target domain similarity

&#x20;       |

&#x20;       v

Jensen-Shannon Divergence

&#x20;       |

&#x20;       v

Select sufficiently similar sources

&#x20;       |

&#x20;       v

Use selected sources for adaptation

```



The goal is to reduce negative transfer from source subjects whose EEG distributions are substantially different from the target.



\---



\## 7. Subdomain-Level Adaptation



The implementation performs domain alignment between source and target feature representations.



The subdomain-level component uses an MMD-based loss.



Conceptually:



```text

Source representation

&#x20;       |

&#x20;       |\\

&#x20;       | \\

&#x20;       |  --> MMD loss

&#x20;       | /

&#x20;       |/

Target representation

```



The objective is to reduce the distribution difference between source and target representations.



\---



\## 8. Category-Level Adaptation



The model also uses prediction information for category-level adaptation.



Source-specific classifiers produce predictions, and confident predictions can be used to obtain pseudo-label information.



The general process is:



```text

Source / target features

&#x20;       |

&#x20;       v

Source-specific classifiers

&#x20;       |

&#x20;       v

Predicted class probabilities

&#x20;       |

&#x20;       v

Confidence filtering

&#x20;       |

&#x20;       v

Pseudo-label information

&#x20;       |

&#x20;       v

Category-level alignment

```



This encourages features belonging to corresponding emotion categories to become more aligned across domains.



\---



\## 9. Training Objective



The overall objective combines classification and domain-adaptation terms:



```text

Total Loss

&#x20;   =

Classification Loss

&#x20;   +

Subdomain Adaptation Loss

&#x20;   +

Category Adaptation Loss

```



The implementation of the individual loss components is contained in:



```text

asjda/losses.py

```



The training procedure is implemented in:



```text

asjda/train.py

```



\---



\## 10. Leave-One-Subject-Out Evaluation



The evaluation uses \*\*Leave-One-Subject-Out (LOSO)\*\* cross-subject evaluation.



For each fold:



```text

One subject â†’ Target



All remaining subjects â†’ Sources

```



For DEAP this produces:



```text

32 LOSO folds

```



For example:



```text

Fold 1:

Target = S01

Sources = S02â€“S32



Fold 2:

Target = S02

Sources = S01, S03â€“S32



...



Fold 32:

Target = S32

Sources = S01â€“S31

```



The target subject is not used as a labeled source during training.



\---



\## 11. Configuration Files



The experiment configurations are stored in:



```text

configs/

â”œâ”€â”€ deap\_valence.yaml

â””â”€â”€ deap\_arousal.yaml

```



The configuration files define the dataset, label mode, training parameters, and experiment settings.



\---



\## 12. Running the Experiments



The LOSO experiment is launched using:



```powershell

python scripts/run\_loso.py

```



The script uses the selected configuration to prepare the source and target domains and perform the LOSO experiment.



\---



\## 13. Experiment Outputs



Experiment outputs are stored in:



```text

runs/

```



The `runs/` directory is excluded from Git because experiment outputs can become large.



Typical outputs include:



\* fold-level predictions

\* target-subject accuracy

\* classification metrics

\* confusion matrices

\* experiment summaries



\---



\## 14. Result Visualization



The repository contains:



```text

scripts/plot\_loso\_results.py

```



This script is used to visualize LOSO experiment results after the experiments have been completed.



\---



\## 15. Current Replication Status



The DEAP replication currently includes:



\* \[x] DEAP dataset preparation

\* \[x] Subject-wise preprocessing

\* \[x] Differential Entropy feature preparation

\* \[x] 160-dimensional DE features

\* \[x] Manifest generation

\* \[x] DEAP data loading

\* \[x] LOSO evaluation

\* \[x] Shared feature extraction

\* \[x] Source-specific feature extraction

\* \[x] Source-specific classification

\* \[x] Domain-adaptation components

\* \[x] Valence experiment

\* \[x] Arousal experiment

\* \[ ] Exact numerical reproduction of every result reported in the paper

\* \[ ] Full reproduction of all baseline methods



Completion of the implementation does not necessarily mean that every reported value from the original paper will be reproduced exactly.



\---



\## 16. Reproducibility



A complete local reproduction should follow this sequence:



```text

1\. Obtain DEAP

&#x20;       |

&#x20;       v

2\. Prepare subject data

&#x20;       |

&#x20;       v

3\. Generate DE features

&#x20;       |

&#x20;       v

4\. Generate manifest

&#x20;       |

&#x20;       v

5\. Select valence or arousal

&#x20;       |

&#x20;       v

6\. Run LOSO

&#x20;       |

&#x20;       v

7\. Save experiment results

&#x20;       |

&#x20;       v

8\. Generate summary statistics

&#x20;       |

&#x20;       v

9\. Plot results

```



Random seeds should be fixed when deterministic or comparable experiments are required.



\---



\## 17. Repository Structure



```text

ASJDA-EEG-Replication/

â”‚

â”œâ”€â”€ README.md

â”œâ”€â”€ requirements.txt

â”‚

â”œâ”€â”€ asjda/

â”‚   â”œâ”€â”€ \_\_init\_\_.py

â”‚   â”œâ”€â”€ config.py

â”‚   â”œâ”€â”€ data.py

â”‚   â”œâ”€â”€ features.py

â”‚   â”œâ”€â”€ losses.py

â”‚   â”œâ”€â”€ model.py

â”‚   â””â”€â”€ train.py

â”‚

â”œâ”€â”€ configs/

â”‚   â”œâ”€â”€ deap\_arousal.yaml

â”‚   â””â”€â”€ deap\_valence.yaml

â”‚

â”œâ”€â”€ scripts/

â”‚   â”œâ”€â”€ prepare\_deap\_dat.py

â”‚   â”œâ”€â”€ download\_kaggle\_deap.py

â”‚   â”œâ”€â”€ run\_loso.py

â”‚   â””â”€â”€ plot\_loso\_results.py

â”‚

â””â”€â”€ docs/

&#x20;   â”œâ”€â”€ README.md

&#x20;   â”œâ”€â”€ architecture.md

&#x20;   â””â”€â”€ ASJDA\_DEAP\_Replication\_Report\_Detailed.md

```



\---



\## 18. Git and Dataset Management



Large dataset files are intentionally excluded from version control.



The repository `.gitignore` excludes:



```text

\*.dat

\*.zip

\*.rar

\*.mat

runs/

```



This keeps the repository lightweight.



The dataset must therefore be obtained and prepared separately before running the experiments.



\---



\## 19. Important Implementation Note



The current implementation is a cleaned DEAP-focused replication.



The repository should be interpreted as an implementation and experimental reproduction of the ASJDA approach rather than a claim that every implementation detail and numerical result is identical to the original authors' environment.



Differences can arise from:



\* preprocessing

\* feature extraction

\* normalization

\* source selection

\* random initialization

\* training settings

\* optimization

\* hardware

\* numerical precision

\* implementation details



Experimental results should therefore always be reported together with the configuration used to obtain them.



\---



\## 20. Reference



\*\*Paper:\*\*

\*Enhancing EEG-Based Cross-Subject Emotion Recognition via Adaptive Source Joint Domain Adaptation (ASJDA)\*



\*\*Journal:\*\*

IEEE Transactions on Affective Computing



The implementation in this repository is intended for academic replication, experimentation, and further development.



```

```
