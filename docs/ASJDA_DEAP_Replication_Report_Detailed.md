# ASJDA–DEAP Paper Replication Report

**Detailed technical summary of dataset preparation, preprocessing, feature extraction, model implementation, arousal-phase work, code inspection, discrepancies, and LOSO evaluation**

## 1. Project Objective

The project focused on an implementation-level replication of the paper “Enhancing EEG-Based Cross-Subject Emotion Recognition via Adaptive Source Joint Domain Adaptation (ASJDA)” using the DEAP dataset. The work was divided into separate valence and arousal phases. The arousal phase covered the complete data preparation and feature-extraction pipeline, adaptation of the reference implementation, source-domain selection, domain-discrepancy components, leave-one-subject-out evaluation, and comparison of the obtained results with the reported reference result.

The objective was an ASJDA-based replication, rather than a claim of exact reproduction. During implementation, differences and shortcomings in the reference repository were identified and corrected where required.

## 2. Dataset Acquisition and Preparation

### DEAP dataset

- Access to the SEED and SEED-IV datasets did not materialize because access is application-gated and no response was received during the project period.
- GAMEEMO was investigated as an openly available alternative, but the dataset structure and processing requirements made it unnecessarily complex for the intended replication, so the work was pivoted to DEAP.
- The official preprocessed DEAP data were downloaded in the `data_preprocessed_python` format, consisting of 32 subject-level `.dat` files.
- The file structure was verified before feature extraction. Each subject contains 40 trials × 40 channels × 8064 samples of EEG data and 40 × 4 rating labels corresponding to valence, arousal, dominance, and liking.
- The experiment was organized as subject-wise domains so that one subject could serve as the target domain while the remaining subjects formed the source pool in LOSO evaluation.

## 3. EEG Preprocessing Pipeline

The preprocessing stage converted the raw/preprocessed DEAP recordings into a consistent representation suitable for feature extraction and cross-subject domain adaptation.

### Channel selection and signal preparation

- The 32 EEG channels used for the feature pipeline were retained for each trial.
- Signal segments were prepared consistently across subjects so that the resulting feature vectors had the same dimensionality.

### Band-pass filtering

- Each EEG channel was filtered into five standard EEG frequency bands before feature extraction.
- The band-wise representation allows the model to capture differences in signal energy and information across frequency ranges rather than treating the complete broadband signal as a single component.

### Epoch/window formation

- The trial recordings were segmented into the windows/epochs required by the feature-extraction pipeline.
- The same segmentation procedure was applied across subjects, ensuring that corresponding feature dimensions represented comparable frequency-band/channel information.

### Normalization and consistency checks

- Preprocessing was applied consistently before feature extraction, with checks performed on dimensions and subject-wise outputs to prevent mismatched input shapes during training.

## 4. Differential Entropy Feature Extraction

Differential Entropy (DE) was used as the EEG feature representation. The completed extraction pipeline was implemented in `extract_deap_features.py` and executed across all 32 subjects.

- For each trial, the 32 EEG channels were passed through the five-band filtering pipeline.
- Differential Entropy was calculated for each channel/frequency-band combination.
- This produced 32 × 5 = 160 features per trial, resulting in a 160-dimensional feature vector.
- With 40 trials per subject, each subject therefore produced approximately a 40 × 160 feature matrix.
- The resulting DE features were stored subject-wise so they could be directly consumed by the ASJDA training and LOSO evaluation code.
- Both valence and arousal labels were extracted and binarized using the rating > 5 threshold. Thus, the arousal phase did not require repeating the feature-extraction stage.

## 5. Arousal-Phase Work Completed

The arousal phase used the completed 160-dimensional DE representation and concentrated specifically on the binary arousal labels. The implementation was organized so that arousal could be selected directly from the training script.

- Arousal labels: extracted and saved for all 32 subjects using the >5 binarization threshold.
- Class-distribution inspection: subject-wise label counts were checked. Strong imbalance was observed in some subjects; for example, subject 26 contained 39 low-arousal and only 1 high-arousal sample.
- Importance of imbalance: extremely skewed target folds can produce unusually high or low accuracy even when the learned decision boundary is not substantially better or worse. This was therefore considered when interpreting subject-wise LOSO accuracy.
- LOSO protocol: each subject was treated as the target/test domain while the remaining subjects supplied source-domain data.
- Arousal execution: the training script was made callable specifically with `--label arousal`.
- Quick verification: a `--test` mode was added to the training script to verify data loading, dimensions, model execution, and evaluation before running the complete 32-subject pass.

## 6. ASJDA Model and Training Implementation

### Reference implementation inspection

- The repository structure and training path were inspected before adapting the code to the DEAP data format.
- The implementation was reorganized into separate model, source-selection, and training components to make the corrected pipeline easier to execute and verify.

### Implemented components

- `model.py`: ASJDA architecture with the corrected multi-kernel MMD implementation.
- `source_selection.py`: corrected JSD-based source selection using PCA followed by histogram-based probability distributions.
- `train_deap.py`: subject-wise LOSO training/evaluation loop with explicit arousal selection through `--label arousal` and a quick `--test` verification mode.
- The adaptation pipeline retained the core ASJDA idea of selecting relevant source subjects and reducing distribution/category discrepancies between source and target domains.

## 7. Discrepancies Found in the Reference Implementation

Inspection of the reference repository revealed two important implementation-level discrepancies. These are relevant because they affect whether the published methodological description and the executable code are actually equivalent.

### Discrepancy 1 — Multi-kernel MMD defined but not used for training

- The repository contains a proper multi-kernel Gaussian MMD implementation, which is closer to the method described for distribution alignment.
- However, the actual training path uses a much simpler linear MMD calculation instead of the defined multi-kernel Gaussian MMD.
- Therefore, the executable training pipeline does not fully correspond to the more sophisticated MMD formulation suggested by the available implementation.
- For the replication, the model code was modified so that the intended multi-kernel MMD component is actually used during training.

### Discrepancy 2 — JSD source selection uses entropy vectors rather than true feature-space distributions

- The reference repository computes JSD using `scipy.stats.entropy` column-wise on 1D feature/entropy vectors.
- This treats those vectors as if they were probability distributions, without first constructing an explicit probability density/discrete distribution over the feature space.
- Consequently, the implementation is a computational shortcut and is not the theoretically clean formulation of Jensen–Shannon divergence between probability distributions representing source and target feature distributions.
- For the replication, JSD source selection was corrected by reducing the feature space with PCA and constructing histogram-based probability distributions before calculating JSD.

**Interpretation:** these corrections mean that the resulting experiments should be described as a corrected/faithful ASJDA-based implementation, not as a byte-for-byte reproduction of the original repository's training behavior.

## 8. Corrected JSD Source Selection

Source selection is a central part of ASJDA because not every source subject is expected to be equally useful for a given target subject. The corrected procedure was designed to compare source and target distributions in a more meaningful probabilistic representation.

- Subject-level DE feature matrices were considered as source/target domains.
- The high-dimensional feature representation was reduced using PCA to make distribution estimation more tractable.
- Histogram-based probability distributions were constructed from the reduced representation.
- Jensen–Shannon divergence was then computed between source and target distributions.
- Sources with smaller distributional divergence were treated as more relevant to the target domain.

## 9. Domain and Category Alignment Components

After source selection, the adaptation stage was evaluated through distribution- and category-level discrepancy components. The implementation investigated the role of MMD and LSD/classifier discrepancy terms in aligning source and target representations.

- **MMD:** measures discrepancy between source and target feature distributions and encourages domain-invariant representations.
- **Multi-kernel MMD correction:** the available Gaussian-kernel formulation was connected to the actual training path rather than leaving it unused.
- **LSD/discrepancy component:** used as part of the category/subdomain alignment mechanism to reduce differences in class-conditional structure.
- These components were considered together with the source-selection stage rather than treating all source subjects as equally informative.

## 10. Leave-One-Subject-Out Evaluation

Cross-subject evaluation was performed using a 32-fold LOSO protocol. For each fold, one subject was held out as the target/test subject and the remaining subjects were available as source domains.

- Target S01 → train/adapt using the remaining subjects.
- Target S02 → train/adapt using the remaining subjects.
- The procedure continued through target S32.
- A subject-wise accuracy was recorded for every target fold.
- The final phase accuracy was calculated as the mean of the 32 subject-wise LOSO accuracies.

## 11. Obtained Results

| Task | Obtained mean LOSO accuracy | Reference DEAP result | Difference |
|---|---:|---:|---:|
| Valence | **68.50%** | 69.31% | **−0.81 pp** |
| Arousal | **76.72%** | 69.31% | **+7.41 pp** |
| Mean of two obtained phases | **72.61%** | 69.31% | **+3.30 pp** |

The reported reference DEAP result is 69.31%. The obtained valence mean of 68.50% is 0.81 percentage points below this value, while the obtained arousal mean of 76.72% is 7.41 percentage points above it. The simple arithmetic mean of the two obtained phase means is 72.61%.

**Important comparison caveat:** the 69.31% value is the paper's reported DEAP result and should not automatically be interpreted as a separate valence or arousal accuracy. Differences in preprocessing, task construction, source/target setup, hyperparameters, and aggregation can make a direct numerical comparison imperfect. The results are therefore best presented as an ASJDA-based replication/implementation comparison.

## 12. Subject-Wise LOSO Performance

The subject-wise LOSO outputs were generated separately for the corresponding phases. The reported valence mean is **68.50%**, and the reported arousal mean is **76.72%**.

## 13. Complete Work Summary

- DEAP dataset acquisition and verification for all 32 subjects.
- Verification of the official preprocessed DEAP `.dat` structure and trial/label dimensions.
- EEG signal preprocessing and preparation.
- Band-pass filtering into five frequency bands.
- Epoch/window preparation for feature extraction.
- Differential Entropy feature extraction for all 32 subjects.
- Generation of 160-dimensional DE feature vectors per trial.
- Extraction and binary thresholding of valence and arousal labels.
- Dedicated arousal-label preparation and class-distribution inspection.
- Construction of subject-wise source and target domains.
- Inspection and adaptation of the reference ASJDA repository.
- Identification of the mismatch between the defined multi-kernel Gaussian MMD and the simpler linear MMD actually used by the training path.
- Identification of the shortcut JSD implementation based on 1D entropy vectors.
- Correction of JSD using PCA and histogram-based probability distributions.
- Correction/integration of the multi-kernel MMD component in the model implementation.
- Integration of arousal selection through `--label arousal`.
- Addition of `--test` mode for quick pipeline verification.
- Execution of 32-subject LOSO evaluation.
- Generation of subject-wise accuracy plots.
- Comparison of obtained valence and arousal performance with the reference DEAP result.

## 14. Interpretation of the Arousal Results

The arousal phase produced a mean LOSO accuracy of **76.72%**. This exceeds the reference DEAP value of 69.31% by 7.41 percentage points. However, the subject-wise results show that performance varies substantially between target subjects. Such variation is expected in cross-subject EEG because individual subjects have different signal characteristics and label distributions.

The strong class imbalance observed in some folds is particularly important. For example, a subject with 39 low-arousal and only 1 high-arousal sample can obtain a high accuracy from a classifier that predominantly predicts the majority class. Therefore, accuracy alone should not be interpreted as definitive evidence of superior class discrimination for every target subject.

The corrected JSD and MMD implementation also changes the computational behavior relative to the executable reference repository. Consequently, the observed performance reflects the corrected ASJDA-based pipeline used for this project rather than an exact reproduction of the original code path.

## 15. Conclusion

The replication workflow progressed from DEAP acquisition and EEG preprocessing through five-band representation, epoch preparation, Differential Entropy extraction, binary arousal-label preparation, adaptive source selection, domain/category discrepancy modeling, and 32-fold LOSO evaluation. All 32 subjects were processed for the DE representation, and the arousal pipeline was made independently executable through the adapted training code.

Two implementation discrepancies in the reference repository were identified and addressed: the defined multi-kernel Gaussian MMD was replaced in the actual training path by a simpler linear MMD, and JSD source selection was implemented on 1D entropy vectors rather than explicit probability distributions. The corrected implementation therefore provides a more methodologically consistent ASJDA-based experiment.

The final mean accuracies were **68.50% for valence** and **76.72% for arousal**. The arousal result is above the paper's reported DEAP result of 69.31%, while the valence result is slightly below it. These results should be interpreted alongside the differences between the reference implementation and the corrected replication pipeline.

## Reference

Ke Liu et al., “Enhancing EEG-Based Cross-Subject Emotion Recognition via Adaptive Source Joint Domain Adaptation (ASJDA),” *IEEE Transactions on Affective Computing*, 2025, DOI: 10.1109/TAFFC.2024.3514635.
