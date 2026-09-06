# Independent Replication Report: ASJDA on the DREAMER Dataset

**Replicated Paper:** Adaptive Source Joint Domain Adaptation for EEG Cross-Subject Emotion Recognition  
**Target Architecture:** Multi-Source Joint Domain Adaptation (Shared MLP + Domain-Specific Feature Extractors & Classifiers)  
**Dataset Targeted:** DREAMER (23 Subjects, 14-Channel Low-Density EEG, Emotiv EPOC)  
**Execution Hardware:** Intel Core i5-12450HX, NVIDIA GeForce RTX 3050 Laptop GPU (6 GB VRAM, CUDA 12.1)  
**Replication Date:** September 2026  

---

## 1. Executive Summary & Experimental Design

This report documents an independent, clean-room replication and methodological audit of the ASJDA framework applied to the DREAMER database. The published paper established performance benchmarks exclusively on laboratory-grade, high-density EEG configurations — SEED (62-channel gelled), SEED-IV (62-channel gelled), and DEAP (32-channel active Biosemi). This replication examines whether ASJDA successfully generalises to consumer-grade, low-density (14-channel) saline EEG.

### Implementation Specifications

- **Signal Preprocessing:** Non-overlapping 2.0-second sliding windows producing 1,861 temporal samples per subject across 18 film trials (42,803 total windows across all 23 participants). Baseline correction was performed by subtracting the pre-trial neutral baseline window means.
- **Feature Representation:** Differential Entropy (DE) computed across five standard physiological frequency bands — $\theta$ (4–8 Hz), $\alpha$ (8–13 Hz), $\beta$ (13–30 Hz), $\gamma_1$ (30–45 Hz), and $\gamma_2$ (45–50 Hz) — constructing a 70-dimensional feature vector ($14\ \text{channels} \times 5\ \text{bands}$) per sample window.
- **Cross-Validation Framework:** Rigorous 23-fold Leave-One-Subject-Out (LOSO) cross-validation. For each fold $T \in \{1, \dots, 23\}$, subject $T$ serves as the completely unlabeled target domain, while the remaining 22 subjects serve as candidate sources.
- **Source Selection Mechanism:** Principal Component Analysis (10 components) and 20-bin histogram estimation were used to establish pairwise Jensen-Shannon Divergence (JSD). Adaptive thresholds $t$ were calibrated per fold using a target-free pseudo-target heuristic.
- **Dynamic Loss Weighting Schedule:** Implemented according to Equation 14 of the published paper:

$$\alpha = \frac{2}{1 + \exp\!\left(-10 \cdot \frac{\text{epoch}}{\text{total\_epochs}}\right)} - 1, \qquad \beta = \frac{\alpha}{100}, \qquad \gamma = \alpha - 1$$

$$\mathcal{L} = \mathcal{L}_{cls} + \alpha \mathcal{L}_{mmd} + \beta \mathcal{L}_{disc} + \gamma \mathcal{L}_{lsd}$$

- **Hardware & Runtime Optimisation:** Resident GPU VRAM preloading and zero-sync asynchronous epsilon scaling ($\epsilon = 10^{-5}$) eliminated over 1,020,000 CPU-to-GPU synchronisation calls per fold, stabilising per-step throughput at roughly 8–12 ms and reducing per-fold training time from approximately 25 minutes to under 2 minutes.

---

## 2. Empirical Benchmark Results

Full 23-fold LOSO cross-validation yielded a nominal classification accuracy of **58.43% ± 8.83%** on Valence and **51.76% ± 9.28%** on Arousal.

### Cross-Subject Benchmark Performance Matrix (Table IV Replica)

| Dataset / Benchmark | Electrode Setup & Hardware | Feature Dims | Valence Accuracy | Arousal Accuracy | Balanced Acc (V / A) |
| :--- | :--- | :---: | :--- | :--- | :--- |
| **SEED (Paper)** | 62 Channels (Gelled Ag/AgCl) | 310-D DE | $87.45\% \pm 8.74\%$ | N/A (3-Class) | $\sim 87.5\%$ |
| **SEED-IV (Paper)** | 62 Channels (Gelled Ag/AgCl) | 310-D DE | $79.58\% \pm 8.23\%$ | N/A (4-Class) | $\sim 79.6\%$ |
| **DEAP (Paper)** | 32 Channels (Active Biosemi) | 160-D DE | $68.30\% \pm 8.93\%$ | $69.31\% \pm 11.26\%$ | $\sim 68.8\%$ |
| **DREAMER (Replicated)** | 14 Channels (Saline Emotiv EPOC) | 70-D DE | **$58.43\% \pm 8.83\%$** | **$51.76\% \pm 9.28\%$** | **$50.77\% / 50.85\%$** |
| **DREAMER Trivial Baseline** | Majority Class Prior (Class 0) | None | **$58.90\%$** | **$52.05\%$** | **$50.00\% / 50.00\%$** |

---

## 3. Failure Mode: Pseudo-Label Confirmation Feedback & Majority Prior Collapse

Evaluating class-balanced accuracy — the arithmetic mean of sensitivity and specificity — reveals that the nominal accuracy numbers do not reflect genuine affective decoding but rather a degenerate collapse into predicting the background majority class.

### Aggregated Confusion Matrix Analysis

**Valence LOSO:**

- Class 0 Accuracy (Low Valence): **93.78%**
- Class 1 Accuracy (High Valence): **7.76%**
- **Balanced Accuracy:** $\dfrac{93.78\% + 7.76\%}{2} = \mathbf{50.77\%}$

**Arousal LOSO:**

- Class 0 Accuracy (Low Arousal): **72.92%**
- Class 1 Accuracy (High Arousal): **28.78%**
- **Balanced Accuracy:** $\dfrac{72.92\% + 28.78\%}{2} = \mathbf{50.85\%}$

### Theoretical Failure Mechanics

1. **Unbalanced Source Priors:** Across all 42,803 extracted windows in DREAMER, the raw ground-truth distributions exhibit an inherent marginal bias toward Class 0 — $58.90\%$ in Valence and $52.05\%$ in Arousal. The multi-source classifiers rapidly absorbed this prior from the source domains during training.

2. **Pseudo-Label Confirmation Loop in LSD:** Local Sub-domain Discrepancy (LSD) calculates alignment using target pseudo-labels:

$$\hat{y}^T = \arg\max \frac{1}{K}\sum_i \hat{y}_i^T$$

Because source heads began optimisation with an initial bias toward Class 0, target samples were overwhelmingly assigned Class 0 pseudo-labels from the start. Equation 13 then forcefully pulled target representations into source Class 0 clusters, penalising any latent activations that attempted to represent Class 1. This positive feedback loop progressively erased the Class 1 decision boundary.

3. **Target Inversion Catastrophes:** In folds where a participant's ground-truth distribution strongly favoured Class 1, the model collapsed well below chance level:
   - **Subject 10 (Valence):** Personal ground truth was 62.3% High Valence; model achieved only **38.21%** accuracy.
   - **Subject 19 (Arousal):** Personal ground truth was 83.9% High Arousal; model achieved only **32.78%** accuracy.

---

## 4. Full Ablation Study (Table VII Replica)

Three systematic ablations were run across all 23 folds on Valence to test the individual contribution of each architectural module:

1. **w/o LSD (`no_lsd`):** Sets $\gamma = 0$, completely disabling category-level sub-domain alignment.
2. **w/o Selection (`no_selection`):** Bypasses adaptive JSD thresholding and fixes $K = 22$ sources for all targets.
3. **w/o MMD (`no_mmd`):** Sets $\alpha = 0$, disabling domain-level feature alignment.

### Empirical Ablation Matrix (Valence LOSO)

| Module Configuration | Evaluated Loss Objective | Mean Accuracy | Std Dev | $\Delta$ vs Baseline | Paired $t$-test | Wilcoxon Signed-Rank |
| :--- | :--- | :---: | :---: | :---: | :--- | :--- |
| **Full ASJDA Baseline** | $\mathcal{L}_{cls} + \alpha \mathcal{L}_{mmd} + \beta \mathcal{L}_{disc} + \gamma \mathcal{L}_{lsd}$ | **58.43%** | $\pm 8.83\%$ | — | — | — |
| **w/o LSD (`no_lsd`)** | $\mathcal{L}_{cls} + \alpha \mathcal{L}_{mmd} + \beta \mathcal{L}_{disc}$ | **59.02%** | $\pm 12.02\%$ | $+0.59\%$ | $t = -0.337,\ p = 0.739$ | $W = 119.0,\ p = 0.580$ |
| **w/o Selection (`no_selection`)** | Full Loss ($K = 22$ sources fixed) | **58.73%** | $\pm 8.46\%$ | $+0.30\%$ | $t = -0.631,\ p = 0.534$ | $W = 122.0,\ p = 0.884$ |
| **w/o MMD (`no_mmd`)** | $\mathcal{L}_{cls} + \beta \mathcal{L}_{disc} + \gamma \mathcal{L}_{lsd}$ | **58.70%** | $\pm 7.94\%$ | $+0.27\%$ | $t = -0.291,\ p = 0.774$ | $W = 129.0,\ p = 0.800$ |
| **Majority Prior Baseline** | Constant Class 0 Prediction | **58.90%** | $\pm 0.00\%$ | $+0.47\%$ | — | — |

### Statistical Hypothesis Testing

A non-parametric **Friedman test** across all four experimental conditions yields $Q = 0.9605$ ($p = 0.8108$). There is no statistically significant difference between the full ASJDA architecture and any ablated variant. Every configuration converged to within $0.6\%$ of the raw dataset prior ($58.90\%$).

---

## 5. Methodological Code Audit: Published Paper vs. Official Repository

An in-depth audit of the authors' official GitHub implementation ([github.com/Pam098/ASJDA](https://github.com/Pam098/ASJDA)) revealed critical discrepancies between the published manuscript and the released code:

1. **MMD Kernel Substitution:** While Section III-C describes a multi-kernel Gaussian RBF formulation using five bandwidths, `model.py` (line 77) calls `utils.mmd_linear()`, executing a linear matrix dot product $\text{Tr}(\Delta \Delta^T)$ instead.

2. **Feature-Space Consensus Discrepancy Bug:** Section III-D defines consensus discrepancy over classifier probability distributions $\hat{y}_i^T \in \mathbb{R}^C$. In `model.py` (lines 80–84), the authors applied `F.softmax()` across the **32-dimensional latent feature vector** `data_tgt_DSFE`, calculating discrepancy over unnormalized latent coordinate dimensions rather than class logits.

3. **Inverted Pseudo-Label Slicing:** In `model.py` (lines 86–90), target feature masking in LSD is derived from `pred_src` (source prediction confidences) rather than target ensemble predictions. Additionally, `lsd.py` (line 66) explicitly hardcodes `num_class = 3`, which causes index errors on binary datasets such as DEAP or DREAMER.

4. **1D Entropy JSD Compression:** The paper defines JSD over subject feature probability distributions. In `main.py` (lines 226–231), the authors computed 1D marginal feature entropy across time using `scipy.stats.entropy`, evaluating JSD over two 1D entropy vectors rather than multivariate feature distributions.

5. **Inner-Loop Optimizer Instantiation:** In `main.py` (line 55), `optimizer = torch.optim.Adam()` is instantiated **inside the inner training loop**, erasing accumulated momentum ($m_t$) and squared gradient ($v_t$) tracking at every step.

---

## 6. Final Replication Verdict

1. **Failure of Transfer on Low-Density Saline EEG:** ASJDA does not achieve positive domain adaptation on the DREAMER dataset. The reported cross-subject accuracies of $58.43\%$ (Valence) and $51.76\%$ (Arousal) match trivial majority-class baselines ($58.90\%$ and $52.05\%$ respectively), yielding near-chance balanced accuracies of $50.77\%$ and $50.85\%$.

2. **Ablation Invariance:** Statistical testing ($p = 0.8108$) establishes that neither MMD, LSD, nor Adaptive JSD source selection actively drove affective decoding on 14-channel saline EEG.

3. **Prerequisites for Consumer EEG Transfer:** To prevent majority collapse on low-density headsets, domain adaptation architectures must incorporate class-balanced focal objectives, dynamic confidence gating on pseudo-labels (threshold $> 0.80$), and covariance Riemannian alignment prior to non-linear feature extraction.
