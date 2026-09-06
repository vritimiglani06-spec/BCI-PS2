# Independent Replication Report: ASJDA on the DREAMER Dataset

**Method Replicated:** Adaptive Source Joint Domain Adaptation for EEG Cross-Subject Emotion Recognition  
**Target Architecture:** Multi-Source Joint Domain Adaptation (Shared MLP + Domain-Specific Extractors & Classifiers)  
**Dataset:** DREAMER (23 Subjects, 14-Channel Low-Density EEG, Emotiv EPOC)  
**Hardware Infrastructure:** Intel Core i5-12450HX, NVIDIA GeForce RTX 3050 Laptop GPU (6GB VRAM), PyTorch 2.3+CUDA 12.1  

---

## 1. Executive Summary & Experimental Configuration

This report details an independent, clean-room replication of the ASJDA methodology applied to the DREAMER EEG database. While the original authors validated ASJDA exclusively on laboratory-grade, high-density setups (SEED 62-channel, SEED-IV 62-channel, DEAP 32-channel), this replication tests cross-subject generalization on consumer-grade, low-density (14-channel) saline EEG.

### Implementation Parameters
* **Feature Extraction:** 2-second non-overlapping sliding windows yielding 1,861 samples per subject (42,803 total windows across 23 subjects). Differential Entropy (DE) extracted across five physiological bands: $\theta$ (4–8 Hz), $\alpha$ (8–13 Hz), $\beta$ (13–30 Hz), $\gamma_1$ (30–45 Hz), and $\gamma_2$ (45–50 Hz), producing a 70-dimensional feature vector per window ($14 \times 5$).
* **Cross-Validation Scheme:** 23-fold Leave-One-Subject-Out (LOSO). In each fold, subject $T$ serves as the unlabeled target, while a subset of remaining subjects $S_k$ are adaptively selected as sources.
* **Optimization & Dynamics:** Adam optimizer ($\text{lr} = 0.001$, $\text{batch\_size} = 32$, 200 epochs per fold). Loss weighting dynamically adjusted via Equation 14:
  $$\alpha = \frac{2}{1 + \exp\left(-10 \cdot \frac{\text{epoch}}{200}\right)} - 1, \quad \beta = \frac{\alpha}{100}, \quad \gamma = \alpha - 1$$
* **Pipeline Acceleration:** Fully vectorized, VRAM-resident tensor slicing with asynchronous denominator epsilon scaling ($\epsilon = 10^{-5}$), reducing per-fold GPU synchronization latency from 25 minutes down to 1.8 minutes.

---

## 2. Empirical Findings & Baseline Comparison

ASJDA achieved a nominal Leave-One-Subject-Out accuracy of **$58.43\% \pm 8.83\%$** on Valence and **$51.76\% \pm 9.28\%$** on Arousal.

### Replication Performance Matrix

| Metric | ASJDA DEAP (Published) | ASJDA DREAMER (Replicated) | Performance Delta | Trivial Prior Baseline |
| :--- | :--- | :--- | :--- | :--- |
| **Valence Accuracy** | $68.30\% \pm 8.93\%$ | **$58.43\% \pm 8.83\%$** | **-9.87%** | $58.90\%$ |
| **Arousal Accuracy** | $69.31\% \pm 11.26\%$ | **$51.76\% \pm 9.28\%$** | **-17.55%** | $52.05\%$ |
| **Valence Balanced Accuracy** | $\sim 68\%$ | **$50.77\%$** | **-17.23%** | $50.00\%$ |
| **Arousal Balanced Accuracy** | $\sim 69\%$ | **$50.85\%$** | **-18.15%** | $50.00\%$ |
| **Channel Count** | 32 (Active Biosemi) | 14 (Saline Emotiv EPOC) | -18 channels | N/A |
| **Feature Dimensionality** | 160-D | 70-D | -90 dimensions | N/A |

---

## 3. Failure Mode: Pseudo-Label Confirmation Feedback & Majority Prior Collapse

Evaluating balanced accuracy reveals that the nominal accuracy numbers are an artifact of class imbalance rather than true affective decoding.
* **Valence Confusion Matrix:** Class 0 (Low): **$93.78\%$** | Class 1 (High): **$7.76\%$** (Balanced Accuracy: **$50.77\%$**).
* **Arousal Confusion Matrix:** Class 0 (Low): **$72.92\%$** | Class 1 (High): **$28.78\%$** (Balanced Accuracy: **$50.85\%$**).

### Root Cause Breakdown
1. **Unbalanced Source Priors:** In DREAMER, raw labels exhibit a natural marginal bias toward Class 0 (58.90% in Valence, 52.05% in Arousal).
2. **Pseudo-Label Confirmation Feedback Loop in LSD:** Local Sub-domain Discrepancy (LSD) uses ensemble pseudo-labels $\hat{y}^T = \arg\max \frac{1}{K}\sum_i \hat{y}_i^T$. Because source classifiers begin training with a slight marginal bias toward Class 0, target pseudo-labels are overwhelmingly predicted as Class 0. Equation 13 then forces target feature representations toward source Class 0 clusters. This penalizes any activation for Class 1, locking the network into a degenerate global majority predictor.
3. **Target Inversion Catastrophes:** In folds where a subject’s true ground-truth distribution was inverted (e.g., S10 in Valence with 62.3% High Valence; S19 in Arousal with 83.9% High Arousal), the model collapsed, achieving **$38.21\%$** on S10 and **$32.78\%$** on S19.

---

## 4. Full Ablation Study (Replicating Table VII)

To evaluate whether performance gains were driven by specific architectural modules, three systematic ablations were executed across all 23 folds for Valence:
1. **w/o LSD (`no_lsd`):** Sets $\gamma = 0$, disabling category-level sub-domain alignment.
2. **w/o Selection (`no_selection`):** Bypasses JSD threshold calibration and fixes $K = 22$ sources for all targets.
3. **w/o MMD (`no_mmd`):** Sets $\alpha = 0$, disabling domain-level feature alignment.

### Empirical Ablation Results

| Module Configuration | Mean Accuracy | Std Dev | $\Delta$ vs Baseline | Paired $t$-test | Wilcoxon Signed-Rank |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Full ASJDA Baseline** | **58.43%** | $\pm 8.83\%$ | — | — | — |
| **w/o LSD (`no_lsd`)** | **59.02%** | $\pm 12.02\%$ | $+0.59\%$ | $t = -0.337, p = 0.739$ | $W = 119.0, p = 0.580$ |
| **w/o Selection (`no_selection`)** | **58.73%** | $\pm 8.46\%$ | $+0.30\%$ | $t = -0.631, p = 0.534$ | $W = 122.0, p = 0.884$ |
| **w/o MMD (`no_mmd`)** | **58.70%** | $\pm 7.94\%$ | $+0.27\%$ | $t = -0.291, p = 0.774$ | $W = 129.0, p = 0.800$ |

**Statistical Significance Assessment:**
A non-parametric **Friedman test** across all four configurations yields $Q = 0.9605$ ($p = 0.8108$). There is zero statistically significant difference between the full ASJDA framework and any ablated variant. Every configuration gravitates to within $0.6\%$ of the $58.90\%$ majority class baseline.

---

## 5. Architectural & Implementation Discrepancies: Paper vs. Official Code

Auditing the authors' official GitHub repository (`github.com/Pam098/ASJDA`) revealed fundamental contradictions between the published text and the operational codebase:

1. **MMD Kernel Substitution:** While Section III-C claims multi-scale Gaussian RBF kernels with five bandwidths, the official `model.py` (line 77) calls `utils.mmd_linear()`, executing a simple linear dot product $\text{Tr}(\Delta \Delta^T)$.
2. **Feature-Space Discrepancy Bug:** Equation 8 defines consensus discrepancy across classifier probability distributions $\hat{y}_i^T \in \mathbb{R}^C$. In `model.py` (lines 80–84), the authors applied `F.softmax()` across the **32-dimensional latent feature vector** `data_tgt_DSFE`, calculating discrepancy over unnormalized feature coordinates rather than class predictions.
3. **Target Pseudo-Labeling Contradiction:** In `model.py` (lines 86–90), target feature masking is calculated using confidence scores from `pred_src` (source predictions) rather than target ensemble predictions, and `lsd.py` (line 66) strictly hardcodes `num_class = 3`.
4. **JSD Dimensionality Compression:** The paper describes distribution divergence across subject feature spaces. In `main.py` (lines 226–231), the authors computed 1D marginal feature entropy across time (`scipy.stats.entropy`), evaluating JSD over two 1D entropy vectors rather than multivariate feature distributions.
5. **Optimizer State Re-initialization:** In `main.py` (line 55), `optimizer = torch.optim.Adam()` is called **inside the inner training loop**, erasing momentum and second-moment buffers every step.

---

## 6. Replication Verdict

1. **Failure of Transfer on Low-Density EEG:** ASJDA fails to achieve positive domain adaptation on the DREAMER dataset. The model's nominal accuracy of $58.43\%$ (Valence) and $51.76\%$ (Arousal) matches a naive majority class predictor ($58.90\%$ and $52.05\%$), yielding near-chance balanced accuracies ($50.77\%$ and $50.85\%$).
2. **Ablation Invariance:** Statistical testing ($p = 0.8108$) demonstrates that neither MMD, LSD, nor Adaptive JSD Source Selection produces measurable transfer on 14-channel saline EEG.
3. **Required Methodological Enhancements:** To adapt ASJDA to low-density consumer headsets, future work must incorporate:
   * Class-balanced focal loss or inverse-frequency cross-entropy to break source prior propagation.
   * Dynamic confidence thresholding ($> 0.80$) and class-balance priors for target pseudo-labeling in LSD.
   * Covariance-based Riemannian alignment prior to feature extraction to combat high saline contact impedance.
