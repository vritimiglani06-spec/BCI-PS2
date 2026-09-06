# src/models.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SharedMLP(nn.Module):
    def __init__(self, input_dim=70):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.net(x)

class DSFE(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32, eps=1e-5, momentum=0.1),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.net(x)

class DSC(nn.Module):
    def __init__(self, n_classes=2):
        super().__init__()
        self.fc = nn.Linear(32, n_classes)

    def forward(self, x):
        return self.fc(x)

def mmd_linear(f_s, f_t):
    """Author's actual implementation from utils.py / model.py (line 77)"""
    delta = f_s - f_t
    return torch.mean(torch.mm(delta, delta.t()))

def compute_disc(probs_list):
    """Vectorized consensus discrepancy across target predictions"""
    K = len(probs_list)
    if K < 2:
        return torch.tensor(0.0, device=probs_list[0].device)
    probs = torch.stack(probs_list, dim=0)  # (K, B, C)
    diff = torch.abs(probs.unsqueeze(1) - probs.unsqueeze(0)).mean(dim=(2, 3))
    triu_idx = torch.triu_indices(K, K, offset=1, device=probs.device)
    return diff[triu_idx[0], triu_idx[1]].mean()

def compute_lsd_fast(source_feat, source_labels, target_feat, target_pseudo_labels, n_classes=2):
    """Zero-sync asynchronous LSD using smooth denominator epsilon"""
    n_s = source_feat.size(0)
    n_t = target_feat.size(0)
    total = torch.cat([source_feat, target_feat], dim=0)

    # Single Gaussian kernel evaluation
    dist_mat = torch.cdist(total, total, p=2) ** 2
    bandwidth = torch.mean(dist_mat.detach()) + 1e-5
    kernel = torch.exp(-dist_mat / bandwidth)

    k_ss = kernel[:n_s, :n_s]
    k_tt = kernel[n_s:, n_s:]
    k_st = kernel[:n_s, n_s:]

    def class_distance(c1, c2):
        s_mask = (source_labels == c1).float()
        t_mask = (target_pseudo_labels == c2).float()
        
        # Pure tensor operations: zero host-device synchronizations
        sw = s_mask / (s_mask.sum() + 1e-5)
        tw = t_mask / (t_mask.sum() + 1e-5)
        
        d1 = torch.sum(torch.outer(sw, sw) * k_ss)
        d2 = torch.sum(torch.outer(tw, tw) * k_tt)
        d3 = torch.sum(torch.outer(sw, tw) * k_st)
        return F.relu(d1 + d2 - 2.0 * d3)

    intra = (class_distance(0, 0) + class_distance(1, 1)) / 2.0
    inter = (class_distance(0, 1) + class_distance(1, 0)) / 2.0
    return intra - inter

    def class_distance(c1, c2):
        s_mask = (source_labels == c1).float()
        t_mask = (target_pseudo_labels == c2).float()
        s_sum, t_sum = s_mask.sum(), t_mask.sum()
        if s_sum == 0 or t_sum == 0:
            return torch.tensor(0.0, device=source_feat.device)
        sw = s_mask / s_sum
        tw = t_mask / t_sum
        d1 = torch.sum(torch.outer(sw, sw) * k_ss)
        d2 = torch.sum(torch.outer(tw, tw) * k_tt)
        d3 = torch.sum(torch.outer(sw, tw) * k_st)
        return F.relu(d1 + d2 - 2.0 * d3)

    intra = (class_distance(0, 0) + class_distance(1, 1)) / 2.0
    inter = (class_distance(0, 1) + class_distance(1, 0)) / 2.0
    return intra - inter

class ASJDA(nn.Module):
    def __init__(self, input_dim=70, n_classes=2, n_sources=3):
        super().__init__()
        self.n_sources = n_sources
        self.n_classes = n_classes
        self.shared_mlp = SharedMLP(input_dim)
        self.dsfe_list = nn.ModuleList([DSFE() for _ in range(n_sources)])
        self.dsc_list = nn.ModuleList([DSC(n_classes) for _ in range(n_sources)])

    def forward(self, source_x_list, source_y_list, target_x):
        K = self.n_sources
        f_com_tgt = self.shared_mlp(target_x)
        f_com_src = [self.shared_mlp(sx) for sx in source_x_list]

        f_spec_src = [self.dsfe_list[i](f_com_src[i]) for i in range(K)]
        f_spec_tgt = [self.dsfe_list[i](f_com_tgt) for i in range(K)]

        logits_src = [self.dsc_list[i](f_spec_src[i]) for i in range(K)]
        logits_tgt = [self.dsc_list[i](f_spec_tgt[i]) for i in range(K)]

        probs_tgt = [F.softmax(lg, dim=1) for lg in logits_tgt]
        mean_tgt_probs = torch.mean(torch.stack(probs_tgt, dim=0), dim=0)
        pseudo_labels = torch.argmax(mean_tgt_probs, dim=1)

        # 1. Classification Loss
        l_cls = torch.mean(torch.stack([
            F.cross_entropy(logits_src[i], source_y_list[i]) for i in range(K)
        ]))

        # 2. Linear MMD Loss (Author's exact implementation)
        l_mmd = torch.mean(torch.stack([
            mmd_linear(f_spec_src[i], f_spec_tgt[i]) for i in range(K)
        ]))

        # 3. Discrepancy Loss
        l_disc = compute_disc(probs_tgt)

        # 4. Fast Sub-domain Discrepancy Loss
        l_lsd = torch.mean(torch.stack([
            compute_lsd_fast(f_spec_src[i], source_y_list[i], f_spec_tgt[i], pseudo_labels, self.n_classes)
            for i in range(K)
        ]))

        return l_cls, l_mmd, l_disc, l_lsd, mean_tgt_probs

    def predict(self, target_x):
        f_com = self.shared_mlp(target_x)
        probs = [F.softmax(self.dsc_list[i](self.dsfe_list[i](f_com)), dim=1) for i in range(self.n_sources)]
        mean_probs = torch.mean(torch.stack(probs, dim=0), dim=0)
        return torch.argmax(mean_probs, dim=1), mean_probs