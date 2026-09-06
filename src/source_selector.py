# src/source_selector.py
import numpy as np
from scipy.stats import entropy
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

class SourceSelector:
    def __init__(self, n_components=10, n_bins=20):
        self.n_components = n_components
        self.n_bins = n_bins
        self.pca = PCA(n_components=self.n_components, random_state=42)
        self.jsd_matrix = None
        self.bin_edges = None
        self.subject_histograms = None

    def fit(self, features_dict):
        """
        Fits PCA on pooled data, projects subjects to 10D,
        and computes the full 23x23 pairwise JSD matrix.
        features_dict: {subj_id (1..23): (N_samples, 70)}
        """
        all_features = np.vstack([features_dict[s] for s in sorted(features_dict.keys())])
        self.pca.fit(all_features)

        # Project and compute shared bin edges per component
        projected = {s: self.pca.transform(features_dict[s]) for s in features_dict}
        all_proj = np.vstack([projected[s] for s in projected])

        self.bin_edges = [
            np.histogram_bin_edges(all_proj[:, d], bins=self.n_bins)
            for d in range(self.n_components)
        ]

        # Calculate normalized probability distributions per subject
        self.subject_histograms = {}
        for s in projected:
            hists = []
            for d in range(self.n_components):
                counts, _ = np.histogram(projected[s][:, d], bins=self.bin_edges[d])
                # Laplace smoothing to prevent zero probabilities
                prob = (counts + 1e-6) / np.sum(counts + 1e-6)
                hists.append(prob)
            self.subject_histograms[s] = np.array(hists)  # (10, 20)

        # Compute full symmetric pairwise JSD matrix
        n = len(features_dict)
        self.jsd_matrix = np.zeros((n, n), dtype=np.float32)
        subj_keys = sorted(features_dict.keys())

        for i in range(n):
            for j in range(i + 1, n):
                s_i, s_j = subj_keys[i], subj_keys[j]
                jsd_val = self._compute_subject_jsd(s_i, s_j)
                self.jsd_matrix[i, j] = jsd_val
                self.jsd_matrix[j, i] = jsd_val

        return self.jsd_matrix

    def _compute_subject_jsd(self, subj_a, subj_b):
        p_all = self.subject_histograms[subj_a]
        q_all = self.subject_histograms[subj_b]
        jsd_dims = []
        for d in range(self.n_components):
            p = p_all[d]
            q = q_all[d]
            m = 0.5 * (p + q)
            # Scipy entropy computes KL divergence
            kl_pm = entropy(p, m)
            kl_qm = entropy(q, m)
            jsd_dims.append(0.5 * (kl_pm + kl_qm))
        return float(np.mean(jsd_dims))

    def calibrate_threshold(self, target_id, features_dict, labels_dict, candidate_thresholds):
        """
        Paper Section III-A:
        1. Find source subject with minimum JSD to target (validation proxy).
        2. Test candidate thresholds on remaining sources using LDA.
        3. Return threshold yielding best proxy decoding accuracy.
        """
        target_idx = target_id - 1
        all_subj = sorted(features_dict.keys())
        source_subjs = [s for s in all_subj if s != target_id]

        # 1. Identify validated proxy subject (minimum JSD to target)
        jsds_to_target = {s: self.jsd_matrix[s - 1, target_idx] for s in source_subjs}
        val_subj = min(jsds_to_target, key=jsds_to_target.get)
        train_subjs = [s for s in source_subjs if s != val_subj]

        X_val = features_dict[val_subj]
        y_val = labels_dict[val_subj]

        best_thresh = candidate_thresholds[0]
        best_acc = -1.0

        # 2. Evaluate candidate thresholds
        for t in candidate_thresholds:
            selected = [s for s in train_subjs if self.jsd_matrix[s - 1, val_subj - 1] < t]
            
            # Require at least 2 sources for training
            if len(selected) < 2:
                continue

            X_tr = np.vstack([features_dict[s] for s in selected])
            y_tr = np.concatenate([labels_dict[s] for s in selected])

            clf = LinearDiscriminantAnalysis()
            clf.fit(X_tr, y_tr)
            acc = clf.score(X_val, y_val)

            if acc > best_acc:
                best_acc = acc
                best_thresh = t

        return best_thresh

    def select_sources(self, target_id, threshold, min_sources=3):
        """
        Returns list of selected source IDs where JSD(source, target) < threshold.
        Clamps to top-min_sources if threshold is too strict.
        """
        target_idx = target_id - 1
        n = self.jsd_matrix.shape[0]
        all_subjs = list(range(1, n + 1))
        source_subjs = [s for s in all_subjs if s != target_id]

        selected = [s for s in source_subjs if self.jsd_matrix[s - 1, target_idx] < threshold]

        # Safety floor: avoid training models on insufficient source domains
        if len(selected) < min_sources:
            sorted_by_jsd = sorted(source_subjs, key=lambda s: self.jsd_matrix[s - 1, target_idx])
            selected = sorted_by_jsd[:min_sources]

        return selected