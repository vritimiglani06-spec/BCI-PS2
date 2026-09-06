# src/run_ablations.py
import math
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from config import DATA_PROCESSED, RESULTS_DIR, DEVICE, SEED
from source_selector import SourceSelector
from models import ASJDA, mmd_linear, compute_disc, compute_lsd_fast

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def get_loss_weights(epoch, total_epochs=200):
    p = float(epoch) / float(total_epochs)
    alpha = 2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0
    beta = alpha / 100.0
    gamma = alpha - 1.0
    return float(alpha), float(beta), float(gamma)

def train_ablation_fold(target_id, selected_sources, feats_gpu, labels_gpu, ablation_mode, n_epochs=200):
    K = len(selected_sources)
    model = ASJDA(input_dim=70, n_classes=2, n_sources=K).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    n_samples = feats_gpu[target_id].size(0)
    batch_size = 32
    n_batches = n_samples // batch_size

    for epoch in range(1, n_epochs + 1):
        model.train()
        alpha, beta, gamma = get_loss_weights(epoch, n_epochs)

        s_perms = [torch.randperm(n_samples, device=DEVICE) for _ in range(K)]
        t_perm = torch.randperm(n_samples, device=DEVICE)

        for b in range(n_batches):
            b_slice = slice(b * batch_size, (b + 1) * batch_size)
            source_x = [feats_gpu[s][s_perms[i][b_slice]] for i, s in enumerate(selected_sources)]
            source_y = [labels_gpu[s][s_perms[i][b_slice]] for i, s in enumerate(selected_sources)]
            target_x = feats_gpu[target_id][t_perm[b_slice]]

            optimizer.zero_grad(set_to_none=True)

            f_com_tgt = model.shared_mlp(target_x)
            f_com_src = [model.shared_mlp(sx) for sx in source_x]

            f_spec_src = [model.dsfe_list[i](f_com_src[i]) for i in range(K)]
            f_spec_tgt = [model.dsfe_list[i](f_com_tgt) for i in range(K)]

            logits_src = [model.dsc_list[i](f_spec_src[i]) for i in range(K)]
            logits_tgt = [model.dsc_list[i](f_spec_tgt[i]) for i in range(K)]

            probs_tgt = [F.softmax(lg, dim=1) for lg in logits_tgt]
            mean_tgt_probs = torch.mean(torch.stack(probs_tgt, dim=0), dim=0)

            # 1. Classification loss
            l_cls = torch.mean(torch.stack([
                F.cross_entropy(logits_src[i], source_y[i]) for i in range(K)
            ]))

            # 2. MMD loss (bypassed if no_mmd)
            if ablation_mode == "no_mmd":
                l_mmd = torch.tensor(0.0, device=DEVICE)
            else:
                l_mmd = torch.mean(torch.stack([
                    mmd_linear(f_spec_src[i], f_spec_tgt[i]) for i in range(K)
                ]))

            # 3. Consensus Discrepancy
            l_disc = compute_disc(probs_tgt)

            # 4. LSD loss (bypassed if no_lsd)
            if ablation_mode == "no_lsd":
                l_lsd = torch.tensor(0.0, device=DEVICE)
            else:
                pseudo_labels = torch.argmax(mean_tgt_probs, dim=1)
                l_lsd = torch.mean(torch.stack([
                    compute_lsd_fast(f_spec_src[i], source_y[i], f_spec_tgt[i], pseudo_labels, model.n_classes)
                    for i in range(K)
                ]))

            total_loss = l_cls + alpha * l_mmd + beta * l_disc + gamma * l_lsd
            total_loss.backward()
            optimizer.step()

        if epoch % 50 == 0:
            print(f"      [Target S{target_id:02d}] Epoch {epoch:03d}/200 complete")

    # Target evaluation
    model.eval()
    with torch.no_grad():
        preds, _ = model.predict(feats_gpu[target_id])
        preds_np = preds.cpu().numpy()
        y_target = labels_gpu[target_id].cpu().numpy()

    acc = float(np.mean(preds_np == y_target) * 100.0)
    del model, optimizer
    torch.cuda.empty_cache()
    return acc

def run_ablation(ablation_mode="no_selection"):
    set_seed(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n================ Commencing Ablation: {ablation_mode.upper()} (VALENCE) ================")

    features_cpu = {s: np.load(DATA_PROCESSED / f"DREAMER_DE_S{s:02d}.npy") for s in range(1, 24)}
    labels_cpu = {s: np.load(DATA_PROCESSED / f"DREAMER_labels_valence_S{s:02d}.npy") for s in range(1, 24)}

    selector = SourceSelector(n_components=10, n_bins=20)
    jsd_matrix = selector.fit(features_cpu)
    upper_tri = jsd_matrix[np.triu_indices(23, k=1)]
    candidates = np.linspace(np.percentile(upper_tri, 20), np.percentile(upper_tri, 85), 10)

    feats_gpu = {s: torch.from_numpy(features_cpu[s]).float().to(DEVICE) for s in range(1, 24)}
    labels_gpu = {s: torch.from_numpy(labels_cpu[s]).long().to(DEVICE) for s in range(1, 24)}

    records = []
    for target in range(1, 24):
        if ablation_mode == "no_selection":
            selected_sources = [s for s in range(1, 24) if s != target]
        else:
            best_t = selector.calibrate_threshold(target, features_cpu, labels_cpu, candidates)
            selected_sources = selector.select_sources(target, best_t)

        print(f"\n---> Starting Fold {target:02d}/23 | Target: S{target:02d} | K={len(selected_sources)} sources")
        acc = train_ablation_fold(target, selected_sources, feats_gpu, labels_gpu, ablation_mode)
        print(f"---> Finished Fold {target:02d}/23 | Target S{target:02d} | Accuracy: {acc:.2f}%")
        records.append({"target_id": target, "accuracy": acc})

    df = pd.DataFrame(records)
    out_path = RESULTS_DIR / f"dreamer_valence_ablation_{ablation_mode}.csv"
    df.to_csv(out_path, index=False)
    print(f"\n{'='*20} Ablation {ablation_mode.upper()} Finished {'='*20}")
    print(f"Final Result: {df['accuracy'].mean():.2f}% +/- {df['accuracy'].std():.2f}%\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="no_selection", choices=["no_lsd", "no_mmd", "no_selection"])
    args = parser.parse_args()
    run_ablation(args.mode)