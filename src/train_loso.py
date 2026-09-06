# src/train_loso.py
import math
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from config import DATA_PROCESSED, CHECKPOINT_DIR, LOG_DIR, RESULTS_DIR, DEVICE, SEED
from source_selector import SourceSelector
from models import ASJDA

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def get_loss_weights(epoch, total_epochs=200):
    """Dynamic parameter tuning (Paper Equation 14)"""
    p = float(epoch) / float(total_epochs)
    alpha = 2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0
    beta = alpha / 100.0
    gamma = alpha - 1.0
    return float(alpha), float(beta), float(gamma)

def train_single_fold(target_id, selected_sources, feats_gpu, labels_gpu, n_epochs=200, label_type="valence"):
    K = len(selected_sources)
    model = ASJDA(input_dim=70, n_classes=2, n_sources=K).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    n_samples = feats_gpu[target_id].size(0)
    batch_size = 32
    n_batches = n_samples // batch_size

    loss_records = []

    for epoch in range(1, n_epochs + 1):
        model.train()
        alpha, beta, gamma = get_loss_weights(epoch, n_epochs)

        # Vectorized on-GPU index shuffling (Zero CPU overhead)
        s_perms = [torch.randperm(n_samples, device=DEVICE) for _ in range(K)]
        t_perm = torch.randperm(n_samples, device=DEVICE)

        ep_cls, ep_mmd, ep_disc, ep_lsd, ep_tot = 0.0, 0.0, 0.0, 0.0, 0.0

        for b in range(n_batches):
            b_slice = slice(b * batch_size, (b + 1) * batch_size)
            
            # Slice directly out of GPU VRAM
            source_x = [feats_gpu[s][s_perms[i][b_slice]] for i, s in enumerate(selected_sources)]
            source_y = [labels_gpu[s][s_perms[i][b_slice]] for i, s in enumerate(selected_sources)]
            target_x = feats_gpu[target_id][t_perm[b_slice]]

            optimizer.zero_grad(set_to_none=True)
            l_cls, l_mmd, l_disc, l_lsd, _ = model(source_x, source_y, target_x)
            total_loss = l_cls + alpha * l_mmd + beta * l_disc + gamma * l_lsd

            total_loss.backward()
            optimizer.step()

            ep_cls += l_cls.item()
            ep_mmd += l_mmd.item()
            ep_disc += l_disc.item()
            ep_lsd += l_lsd.item()
            ep_tot += total_loss.item()

        loss_records.append({
            "epoch": epoch,
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "L_cls": ep_cls / n_batches,
            "L_mmd": ep_mmd / n_batches,
            "L_disc": ep_disc / n_batches,
            "L_lsd": ep_lsd / n_batches,
            "total_loss": ep_tot / n_batches,
        })

    # Save loss logs
    pd.DataFrame(loss_records).to_csv(LOG_DIR / f"{label_type}_fold_{target_id:02d}_loss.csv", index=False)

    # Evaluation on full target set
    model.eval()
    with torch.no_grad():
        preds, probs = model.predict(feats_gpu[target_id])
        preds_np = preds.cpu().numpy()
        probs_np = probs.cpu().numpy()
        y_target = labels_gpu[target_id].cpu().numpy()

    accuracy = float(np.mean(preds_np == y_target) * 100.0)

    # Save predictions
    pd.DataFrame({
        "sample_id": np.arange(len(y_target)),
        "true_label": y_target,
        "predicted_label": preds_np,
        "prob_class_0": probs_np[:, 0],
        "prob_class_1": probs_np[:, 1],
    }).to_csv(LOG_DIR / f"{label_type}_fold_{target_id:02d}_predictions.csv", index=False)

    # Save checkpoint
    torch.save(model.state_dict(), CHECKPOINT_DIR / f"{label_type}_fold_{target_id:02d}.pt")

    del model, optimizer
    torch.cuda.empty_cache()
    return accuracy

def run_loso(label_type="valence", n_epochs=200, target_only=None):
    set_seed(SEED)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading features and {label_type.upper()} labels into Host Memory...")
    features_cpu = {s: np.load(DATA_PROCESSED / f"DREAMER_DE_S{s:02d}.npy") for s in range(1, 24)}
    labels_cpu = {s: np.load(DATA_PROCESSED / f"DREAMER_labels_{label_type}_S{s:02d}.npy") for s in range(1, 24)}

    # Adaptive source selection setup (PCA on CPU)
    selector = SourceSelector(n_components=10, n_bins=20)
    print("Fitting PCA and computing pairwise JSD matrix...")
    jsd_matrix = selector.fit(features_cpu)
    upper_tri = jsd_matrix[np.triu_indices(23, k=1)]
    candidates = np.linspace(np.percentile(upper_tri, 20), np.percentile(upper_tri, 85), 10)

    # PRELOAD ALL DATA DIRECTLY INTO GPU VRAM
    print(f"Transferring all 23 subjects to GPU ({DEVICE})...")
    feats_gpu = {s: torch.from_numpy(features_cpu[s]).float().to(DEVICE) for s in range(1, 24)}
    labels_gpu = {s: torch.from_numpy(labels_cpu[s]).long().to(DEVICE) for s in range(1, 24)}
    print("VRAM resident initialization complete.")

    target_range = [target_only] if target_only else list(range(1, 24))
    summary_records = []

    print(f"\n{'='*20} Commencing {label_type.upper()} LOSO CV ({len(target_range)} folds) {'='*20}")
    for target in target_range:
        best_t = selector.calibrate_threshold(target, features_cpu, labels_cpu, candidates)
        selected_sources = selector.select_sources(target, best_t)

        print(f"\nFold {target:02d}/23 -> Target: S{target:02d} | Calibrated t={best_t:.4f} | K={len(selected_sources)} sources")
        acc = train_single_fold(
            target_id=target,
            selected_sources=selected_sources,
            feats_gpu=feats_gpu,
            labels_gpu=labels_gpu,
            n_epochs=n_epochs,
            label_type=label_type,
        )
        print(f"Fold {target:02d}/23 Result -> Accuracy: {acc:.2f}%")

        summary_records.append({
            "target_id": target,
            "threshold": best_t,
            "n_sources_selected": len(selected_sources),
            "test_accuracy": acc,
        })

    df_summary = pd.DataFrame(summary_records)
    out_file = RESULTS_DIR / f"dreamer_{label_type}_loso_summary.csv"
    df_summary.to_csv(out_file, index=False)

    mean_acc = df_summary["test_accuracy"].mean()
    std_acc = df_summary["test_accuracy"].std()
    print(f"\n{'='*20} LOSO COMPLETE ({label_type.upper()}) {'='*20}")
    print(f"Mean Accuracy: {mean_acc:.2f}% +/- {std_acc:.2f}%")
    print(f"Saved summary to: {out_file}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASJDA Leave-One-Subject-Out Cross Validation")
    parser.add_argument("--label", type=str, default="valence", choices=["valence", "arousal"])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--target", type=int, default=None, help="Run single target subject")
    args = parser.parse_args()

    run_loso(label_type=args.label, n_epochs=args.epochs, target_only=args.target)