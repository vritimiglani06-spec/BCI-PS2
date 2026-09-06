from __future__ import annotations

from itertools import cycle
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix

from .data import SubjectData, make_loader, standardize_sources_and_target
from .losses import (
    classifier_discrepancy,
    dynamic_weights,
    js_divergence,
    local_subdomain_discrepancy,
    mmd_loss,
)
from .model import ASJDA


def select_sources_by_jsd(
    source_subjects: List[SubjectData],
    target_subject: SubjectData,
    threshold: float,
) -> Tuple[List[SubjectData], dict[str, float]]:
    scores = {
        s.subject_id: js_divergence(s.x, target_subject.x)
        for s in source_subjects
    }
    selected = [s for s in source_subjects if scores[s.subject_id] < threshold]
    if not selected:
        best = min(source_subjects, key=lambda s: scores[s.subject_id])
        selected = [best]
    return selected, scores


def train_one_fold(
    source_subjects: List[SubjectData],
    target_subject: SubjectData,
    *,
    input_dim: int,
    num_classes: int,
    jsd_threshold: float,
    batch_size: int = 32,
    epochs: int = 200,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    kernel_mul: float = 2.0,
    kernel_num: int = 5,
    device: str | torch.device = "cpu",
) -> dict:
    selected, jsd_scores = select_sources_by_jsd(source_subjects, target_subject, jsd_threshold)
    selected, target_subject, scaler = standardize_sources_and_target(selected, target_subject)

    model = ASJDA(input_dim, num_classes, len(selected)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    source_loaders = [
        make_loader(s.x, s.y, batch_size, shuffle=True)
        for s in selected
    ]
    target_loader = make_loader(target_subject.x, None, batch_size, shuffle=True)
    steps_per_epoch = max(len(loader) for loader in source_loaders + [target_loader])

    history = []
    for epoch in range(epochs):
        model.train()
        alpha, beta, gamma = dynamic_weights(epoch, epochs)
        source_iters = [cycle(loader) for loader in source_loaders]
        target_iter = cycle(target_loader)

        totals = {"loss": 0.0, "cls": 0.0, "mmd": 0.0, "disc": 0.0, "lsd": 0.0}
        for _ in range(steps_per_epoch):
            source_batches = [next(it) for it in source_iters]
            target_x = next(target_iter).to(device)
            xs = [batch[0].to(device) for batch in source_batches]
            ys = [batch[1].to(device) for batch in source_batches]

            out = model.forward_source_target(xs, target_x)

            cls = sum(F.cross_entropy(logits, y) for logits, y in zip(out["source_logits"], ys)) / len(ys)
            mmd = sum(
                mmd_loss(s_feat, t_feat, kernel_mul, kernel_num)
                for s_feat, t_feat in zip(out["source_specific"], out["target_specific"])
            ) / len(ys)
            disc = classifier_discrepancy(out["target_logits"])

            lsd_terms = []
            for s_feat, y, t_feat, t_logits in zip(
                out["source_specific"], ys, out["target_specific"], out["target_logits"]
            ):
                pseudo = torch.argmax(t_logits.detach(), dim=1)
                lsd_terms.append(
                    local_subdomain_discrepancy(
                        s_feat, y, t_feat, pseudo, num_classes, kernel_mul, kernel_num
                    )
                )
            lsd = torch.stack(lsd_terms).mean()

            loss = cls + alpha * mmd + beta * disc + gamma * lsd
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            for key, value in [("loss", loss), ("cls", cls), ("mmd", mmd), ("disc", disc), ("lsd", lsd)]:
                totals[key] += float(value.detach().cpu())

        history.append({k: v / steps_per_epoch for k, v in totals.items()} | {
            "epoch": epoch + 1,
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
        })

    y_true, y_pred = predict_subject(model, target_subject, batch_size, device)
    return {
        "model": model,
        "scaler": scaler,
        "selected_sources": [s.subject_id for s in selected],
        "jsd_scores": jsd_scores,
        "history": history,
        "accuracy": accuracy_score(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=np.arange(num_classes)),
        "y_true": y_true,
        "y_pred": y_pred,
    }


@torch.no_grad()
def predict_subject(model: ASJDA, subject: SubjectData, batch_size: int, device: str | torch.device):
    model.eval()
    loader = make_loader(subject.x, subject.y, batch_size, shuffle=False)
    y_true = []
    y_pred = []
    for x, y in loader:
        probs = model.predict(x.to(device))
        y_true.extend(y.numpy().tolist())
        y_pred.extend(torch.argmax(probs, dim=1).cpu().numpy().tolist())
    return np.asarray(y_true), np.asarray(y_pred)
