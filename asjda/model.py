from __future__ import annotations

import torch
import torch.nn as nn


class SharedFeatureExtractor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = 64, slope: float = 0.01):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(slope),
            nn.Linear(hidden_dim, 128),
            nn.LeakyReLU(slope),
            nn.Linear(128, output_dim),
            nn.LeakyReLU(slope),
        )

    def forward(self, x):
        return self.net(x)


class DomainSpecificFeatureExtractor(nn.Module):
    def __init__(self, input_dim: int = 64, output_dim: int = 32, slope: float = 0.01):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.LeakyReLU(slope),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ASJDA(nn.Module):
    """ASJDA network with one branch and classifier per selected source."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        num_sources: int,
        shared_hidden: int = 256,
        shared_dim: int = 64,
        specific_dim: int = 32,
        slope: float = 0.01,
    ):
        super().__init__()
        if num_sources < 1:
            raise ValueError("ASJDA needs at least one selected source domain.")
        self.num_sources = num_sources
        self.num_classes = num_classes
        self.shared = SharedFeatureExtractor(input_dim, shared_hidden, shared_dim, slope)
        self.specific = nn.ModuleList([
            DomainSpecificFeatureExtractor(shared_dim, specific_dim, slope)
            for _ in range(num_sources)
        ])
        self.classifiers = nn.ModuleList([
            nn.Linear(specific_dim, num_classes)
            for _ in range(num_sources)
        ])

    def forward_source_target(self, source_batches: list[torch.Tensor], target_x: torch.Tensor):
        target_common = self.shared(target_x)
        source_specific = []
        target_specific = []
        source_logits = []
        target_logits = []

        for idx, source_x in enumerate(source_batches):
            source_common = self.shared(source_x)
            s_feat = self.specific[idx](source_common)
            t_feat = self.specific[idx](target_common)
            source_specific.append(s_feat)
            target_specific.append(t_feat)
            source_logits.append(self.classifiers[idx](s_feat))
            target_logits.append(self.classifiers[idx](t_feat))

        return {
            "source_specific": source_specific,
            "target_specific": target_specific,
            "source_logits": source_logits,
            "target_logits": target_logits,
        }

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        common = self.shared(x)
        probs = []
        for branch, classifier in zip(self.specific, self.classifiers):
            probs.append(torch.softmax(classifier(branch(common)), dim=1))
        return torch.stack(probs, dim=0).mean(dim=0)
