# Methodological Limits of Adaptive Source Domain Adaptation on Consumer-Grade EEG: An Empirical Replication and Failure Mode Analysis of ASJDA

**Author:** Hidanshu Singh  
**Affiliation:** Undergraduate BCI Research Group  
**Target Architecture:** Adaptive Source Joint Domain Adaptation (ASJDA)  
**Target Dataset:** DREAMER (14-Channel Low-Density Saline EEG)  
**Date:** September 2026  

---

## Abstract

Cross-subject emotion recognition in Brain-Computer Interfaces (BCIs) remains constrained by physiological non-stationarity and inter-subject distribution shifts. The Adaptive Source Joint Domain Adaptation (ASJDA) framework addresses this by combining unsupervised source selection via Jensen-Shannon Divergence (JSD) with domain-level (MMD), classifier-consensus (DISC), and category-level (LSD) distribution alignment. While the authors reported cross-subject accuracies of $87.45\%$ on SEED (62 channels) and $68.30\%$ on DEAP (32 channels), this paper tests the hypothesis that ASJDA's domain transfer assumptions break down on low-density, consumer-grade EEG hardware.

Evaluating ASJDA across a 23-fold Leave-One-Subject-Out (LOSO) cross-validation scheme on the DREAMER dataset (14 channels, Emotiv EPOC) yields nominal accuracies of $58.43\% \pm 8.83\%$ (Valence) and $51.76\% \pm 9.28\%$ (Arousal). However, class-balanced accuracy hovers at chance level ($50.77\%$ and $50.85\%$). Systematic ablations demonstrate that disabling MMD, LSD, or Adaptive Source Selection yields statistically indistinguishable results (Friedman test $Q = 0.9605,\ p = 0.8108$), with all configurations converging to the dataset's background majority class prior ($58.90\%$). We identify an unchecked pseudo-label confirmation feedback loop in the LSD module as the primary failure mode and detail five fundamental discrepancies between the published mathematical specification and the authors' released codebase.

---

## 1. Introduction & Theoretical Background

Affective Brain-Computer Interfaces aim to decode human emotion from electroencephalographic (EEG) dynamics. A persistent obstacle to clinical or consumer deployment is cross-subject variance: electrode contact impedance, skull thickness variations, baseline psychological states, and cortical geometry together create severe non-linear covariate shift across individuals.

To bridge this gap, domain adaptation (DA) techniques map subject distributions into domain-invariant latent spaces. ASJDA (*IEEE Transactions on Affective Computing*) introduced a tripartite transfer architecture:

1. **Adaptive Source Selection:** Pruning distant source domains using Jensen-Shannon Divergence (JSD) on latent feature manifolds to mitigate negative transfer.
2. **Global Domain Alignment:** Minimizing Maximum Mean Discrepancy (MMD) between common feature representations.
3. **Category-Level Sub-domain Discrepancy (LSD):** Enforcing intra-class compactness and inter-class separation between source domains and unlabeled target domains using ensemble pseudo-labels.

### The Research Question

Published BCI benchmarks overwhelmingly rely on research-grade caps featuring 32 to 62 active wet or gelled electrodes (e.g., Biosemi ActiveTwo, Neuroscan). Consumer neurotechnology, however, depends on low-density saline or dry headsets (e.g., Emotiv EPOC, 14 channels) with lower signal-to-noise ratios (SNR). The central question this paper addresses is whether ASJDA's transfer mechanism successfully decodes affective states under hardware-constrained, low-density EEG, or whether the loss formulation collapses when exposed to higher channel noise.

---

## 2. Experimental Methodology

### 2.1 Preprocessing and Feature Extraction

The DREAMER dataset consists of 23 participants monitored across 18 audiovisual emotion-elicitation trials (totalling 42,803 seconds of data).

- **Temporal Segmentation:** Continuous EEG signals were segmented into non-overlapping 2.0-second sliding windows, producing exactly 1,861 temporal windows per subject.
- **Baseline Correction:** Pre-stimulus neutral baseline windows were subtracted from trial windows to isolate stimulus-driven spectral perturbations.
- **Differential Entropy (DE):** For a Gaussian-distributed signal segment with variance $\sigma^2$, DE is computed as:

$$h(X) = \frac{1}{2}\ln(2\pi e \sigma^2)$$

DE was extracted across five physiological bands: $\theta$ (4–8 Hz), $\alpha$ (8–13 Hz), $\beta$ (13–30 Hz), $\gamma_1$ (30–45 Hz), and $\gamma_2$ (45–50 Hz). Across 14 channels, this produces a 70-dimensional feature vector per sample window ($14 \times 5$).

### 2.2 Model Architecture

The network consists of a shared common feature extractor ($\text{CFE}: \mathbb{R}^{70} \to \mathbb{R}^{64}$), $K$ domain-specific feature extractors ($\text{DSFE}_k: \mathbb{R}^{64} \to \mathbb{R}^{32}$), and $K$ domain-specific linear classifiers ($\text{DSC}_k: \mathbb{R}^{32} \to \mathbb{R}^2$).

Optimization follows a dynamic weighting schedule across 200 epochs:

$$\mathcal{L} = \mathcal{L}_{cls} + \alpha \mathcal{L}_{mmd} + \beta \mathcal{L}_{disc} + \gamma \mathcal{L}_{lsd}$$

where $\alpha = \frac{2}{1 + \exp(-10p)} - 1$, $\beta = \frac{\alpha}{100}$, $\gamma = \alpha - 1$, and $p = \frac{\text{epoch}}{\text{total\_epochs}}$.

---

## 3. Empirical Results & Comparative Benchmarks

Testing followed a 23-fold Leave-One-Subject-Out (LOSO) structure where each subject served once as the target domain.

### Primary Benchmark Comparison (Table IV Equivalent)

| Dataset | Hardware Configuration | Channels | Dimensions | Valence Acc | Arousal Acc | Balanced Acc (V / A) |
| :--- | :--- | :---: | :---: | :--- | :--- | :--- |
| **SEED (Paper)** | Research Gelled Ag/AgCl | 62 | 310-D | $87.45\% \pm 8.74\%$ | N/A | $\sim 87.5\%$ |
| **DEAP (Paper)** | Active Biosemi | 32 | 160-D | $68.30\% \pm 8.93\%$ | $69.31\% \pm 11.26\%$ | $\sim 68.8\%$ |
| **DREAMER (Ours)** | Consumer Emotiv Saline | 14 | 70-D | **$58.43\% \pm 8.83\%$** | **$51.76\% \pm 9.28\%$** | **$50.77\% / 50.85\%$** |
| **Trivial Prior Baseline** | Constant Class 0 Prediction | 0 | 0 | **$58.90\%$** | **$52.05\%$** | **$50.00\% / 50.00\%$** |

Across all 42,803 windows in DREAMER, the raw ground-truth distributions exhibit a marginal bias toward Class 0 ($58.90\%$ in Valence, $52.05\%$ in Arousal). The nominal accuracy achieved by ASJDA matches this trivial majority prior baseline to within $0.5\%$.

---

## 4. Ablation Analysis & Statistical Hypothesis Testing

To determine whether individual components contributed meaningfully to affective decoding, systematic ablations were conducted across all 23 folds on Valence.

### Ablation Matrix (Table VII Equivalent)

| Variant | Evaluated Objective | Mean Acc | Std Dev | $\Delta$ vs Baseline | Paired $t$-test | Wilcoxon Signed-Rank |
| :--- | :--- | :---: | :---: | :---: | :--- | :--- |
| **Full ASJDA** | Full Loss Schedule | **58.43%** | $\pm 8.83\%$ | — | — | — |
| **w/o LSD** | $\gamma = 0$ (No Category Alignment) | **59.02%** | $\pm 12.02\%$ | $+0.59\%$ | $t = -0.337,\ p = 0.739$ | $W = 119.0,\ p = 0.580$ |
| **w/o Selection** | $K = 22$ (No Source Pruning) | **58.73%** | $\pm 8.46\%$ | $+0.30\%$ | $t = -0.631,\ p = 0.534$ | $W = 122.0,\ p = 0.884$ |
| **w/o MMD** | $\alpha = 0$ (No Domain Alignment) | **58.70%** | $\pm 7.94\%$ | $+0.27\%$ | $t = -0.291,\ p = 0.774$ | $W = 129.0,\ p = 0.800$ |
| **Majority Baseline** | Argmax Marginal Prior | **58.90%** | $\pm 0.00\%$ | $+0.47\%$ | — | — |

A non-parametric **Friedman test** across the four configurations yields $Q = 0.9605$ ($p = 0.8108$). The null hypothesis cannot be rejected: **no individual architectural component produces a statistically significant performance difference over any other variant on this dataset**.

---

## 5. Diagnostic Failure Mode Analysis

Evaluating sensitivity, specificity, and confusion matrix distributions demonstrates why nominal accuracy remained stable while affective decoding effectively failed.

### 5.1 Confirmation Bias in Local Sub-domain Discrepancy (LSD)

Equation 13 aligns source and target clusters using pseudo-labels generated by the ensemble:

$$\hat{y}^T = \arg\max \frac{1}{K}\sum_{k=1}^K \hat{y}_k^T$$

Because the source domains possess an inherent ~59% Class 0 prior, the initial unadapted classifiers output Class 0 probabilities with slightly higher confidence. The pseudo-labeling mechanism consequently labels the majority of target samples as Class 0. LSD then computes the intra-class kernel distances ($e_{tt},\ e_{st}$) and pulls target representations directly into the source Class 0 cluster. This creates an unconstrained confirmation feedback loop:

1. Target samples near the decision boundary are pseudo-labeled as Class 0.
2. LSD penalizes target samples that do not match the Class 0 source distribution.
3. Feature representations for Class 1 are progressively suppressed.
4. By epoch 50, the classifier outputs Class 0 for more than 90% of samples.

This is empirically verified in the aggregated confusion matrices:

- **Valence:** Class 0 Sensitivity = **93.78%**, Class 1 Specificity = **7.76%** (Balanced Acc = **50.77%**)
- **Arousal:** Class 0 Sensitivity = **72.92%**, Class 1 Specificity = **28.78%** (Balanced Acc = **50.85%**)

### 5.2 The Prior Reversal Anomaly

The confirmation loop is further demonstrated by evaluating outlier folds where a subject's true personal distribution strongly favoured Class 1:

- **Subject 10 (Valence):** True distribution was 62.3% High Valence. Because the model predicted Class 0, test accuracy collapsed to **38.21%**.
- **Subject 19 (Arousal):** True distribution was 83.9% High Arousal. The model achieved only **32.78%**.

---

## 6. Codebase Audit: Published Paper vs. Official Repository

Auditing the authors' official GitHub implementation ([github.com/Pam098/ASJDA](https://github.com/Pam098/ASJDA)) uncovered five critical divergences from the published theory:

1. **MMD Formulation Shortcut:** Section III-C explicitly defines multi-kernel Gaussian RBF MMD across five scales. In `model.py` (line 77), the authors bypassed this entirely and called `utils.mmd_linear()` — a linear matrix dot product $\text{Tr}(\Delta \Delta^T)$.

2. **Softmax Applied to Latent Feature Space:** Section III-D defines consensus discrepancy over classifier probability outputs $\hat{y}_i^T \in \mathbb{R}^C$. In `model.py` (lines 80–84), the authors applied `F.softmax()` directly across the **32-dimensional latent feature vector** `data_tgt_DSFE`. This computes discrepancy across arbitrary latent dimensions rather than class probabilities.

3. **Pseudo-Label Masking Error:** In `model.py` (lines 86–90), target feature masking inside LSD is indexed using confidence scores from `pred_src` (source domain predictions) rather than target ensemble predictions. In `lsd.py` (line 66), `num_class = 3` is hardcoded, causing index errors on binary datasets like DEAP or DREAMER.

4. **1D Entropy Compression for JSD:** In `main.py` (lines 226–231), JSD is not computed across multivariate sample space distributions. The code calculates 1D feature-wise entropy across time via `scipy.stats.entropy`, evaluating JSD over two 1D vectors rather than the full feature distributions.

5. **Inner-Loop Optimizer Reset:** In `main.py` (line 55), `optimizer = torch.optim.Adam()` is called inside the iteration loop, erasing accumulated momentum and second-moment gradient tracking at every step.

---

## 7. Conclusions & Guidelines for Consumer BCI Transfer

1. **Replication Verdict:** ASJDA's reported performance advantages do not transfer to 14-channel consumer saline EEG. Nominal classification accuracy matches the marginal class prior ($58.43\% \approx 58.90\%$), while balanced accuracy hovers at chance ($50.77\%$).

2. **Ablation Invariance:** Statistical testing ($p = 0.8108$) confirms that neither MMD, LSD, nor Adaptive JSD source selection actively drove affective decoding on DREAMER.

3. **Necessary Architecture Adaptations for Low-Density EEG:**
   - **Class-Balanced Objectives:** Implement focal loss or inverse-frequency cross-entropy to prevent source priors from leaking into target pseudo-labels.
   - **Confidence Thresholding:** Apply confidence gating (e.g., $p_{\text{max}} \ge 0.80$) on target pseudo-labels before computing category alignment.
   - **Geometric Pre-Alignment:** Introduce Riemannian manifold alignment (e.g., Euclidean alignment of covariance matrices) prior to feature extraction to reduce saline contact impedance drift across sessions.

---

## References

1. Paper Authors, "Adaptive Source Joint Domain Adaptation for EEG Cross-subject Emotion Recognition," *IEEE Transactions on Affective Computing*, 2024.
2. H. Katsigiannis and N. Ramzan, "DREAMER: A Database for Emotion Recognition Through EEG and ECG Signals from Wireless Low-cost Off-the-Shelf Devices," *IEEE Journal of Biomedical and Health Informatics*, 2018.
3. Official ASJDA GitHub Repository: [https://github.com/Pam098/ASJDA](https://github.com/Pam098/ASJDA)
