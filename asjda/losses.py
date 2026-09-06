from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def js_divergence(x_source: np.ndarray, x_target: np.ndarray, bins: int = 64, eps: float = 1e-12) -> float:
    """Jensen-Shannon divergence between flattened feature distributions."""
    lo = min(float(np.min(x_source)), float(np.min(x_target)))
    hi = max(float(np.max(x_source)), float(np.max(x_target)))
    if lo == hi:
        return 0.0
    p, _ = np.histogram(x_source.ravel(), bins=bins, range=(lo, hi), density=False)
    q, _ = np.histogram(x_target.ravel(), bins=bins, range=(lo, hi), density=False)
    p = p.astype(np.float64) + eps
    q = q.astype(np.float64) + eps
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * (np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m))))


def gaussian_kernel(source: torch.Tensor, target: torch.Tensor, kernel_mul: float = 2.0, kernel_num: int = 5):
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0)
    total1 = total.unsqueeze(1)
    l2_distance = ((total0 - total1) ** 2).sum(2)

    n_samples = total.shape[0]
    bandwidth = torch.sum(l2_distance.detach()) / max(n_samples**2 - n_samples, 1)
    bandwidth = bandwidth / (kernel_mul ** (kernel_num // 2))
    bandwidths = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    return sum(torch.exp(-l2_distance / bw.clamp_min(1e-8)) for bw in bandwidths)


def mmd_loss(source: torch.Tensor, target: torch.Tensor, kernel_mul: float = 2.0, kernel_num: int = 5):
    batch_size = min(source.shape[0], target.shape[0])
    source = source[:batch_size]
    target = target[:batch_size]
    kernels = gaussian_kernel(source, target, kernel_mul, kernel_num)
    xx = kernels[:batch_size, :batch_size]
    yy = kernels[batch_size:, batch_size:]
    xy = kernels[:batch_size, batch_size:]
    yx = kernels[batch_size:, :batch_size]
    return torch.mean(xx + yy - xy - yx)


def classifier_discrepancy(target_logits: list[torch.Tensor]) -> torch.Tensor:
    """Mean absolute disagreement between target classifier probabilities."""
    if len(target_logits) < 2:
        return target_logits[0].new_tensor(0.0)
    probs = [F.softmax(logits, dim=1) for logits in target_logits]
    loss = probs[0].new_tensor(0.0)
    pairs = 0
    for i in range(len(probs)):
        for j in range(i + 1, len(probs)):
            loss = loss + torch.mean(torch.abs(probs[i] - probs[j]))
            pairs += 1
    return loss / pairs


def local_subdomain_discrepancy(
    source_features: torch.Tensor,
    source_labels: torch.Tensor,
    target_features: torch.Tensor,
    target_pseudo_labels: torch.Tensor,
    num_classes: int,
    kernel_mul: float = 2.0,
    kernel_num: int = 5,
) -> torch.Tensor:
    """LSD approximation: same-class MMD minus mean different-class MMD."""
    same_terms = []
    diff_terms = []

    for c1 in range(num_classes):
        s_mask = source_labels == c1
        t_same = target_pseudo_labels == c1
        if s_mask.any() and t_same.any():
            same_terms.append(mmd_loss(source_features[s_mask], target_features[t_same], kernel_mul, kernel_num))

        for c2 in range(num_classes):
            if c1 == c2:
                continue
            t_diff = target_pseudo_labels == c2
            if s_mask.any() and t_diff.any():
                diff_terms.append(mmd_loss(source_features[s_mask], target_features[t_diff], kernel_mul, kernel_num))

    base = source_features.new_tensor(0.0)
    same = torch.stack(same_terms).mean() if same_terms else base
    diff = torch.stack(diff_terms).mean() if diff_terms else base
    return same - diff


def dynamic_weights(epoch: int, total_epochs: int):
    progress = float(epoch + 1) / max(float(total_epochs), 1.0)
    alpha = 2.0 / (1.0 + np.exp(-10.0 * progress)) - 1.0
    beta = alpha / 100.0
    gamma = alpha - 1.0
    return float(alpha), float(beta), float(gamma)
