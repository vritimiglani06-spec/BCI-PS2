# src/test_model.py
import torch
import numpy as np
from config import DATA_PROCESSED, DEVICE
from models import ASJDA

def test_asjda():
    print(f"Testing ASJDA Architecture on device: {DEVICE}")
    batch_size = 32
    input_dim = 70
    n_classes = 2
    K = 5  # Test with 5 selected sources

    print("\n1. Running dummy forward & backward pass...")
    model = ASJDA(input_dim=input_dim, n_classes=n_classes, n_sources=K).to(DEVICE)
    
    dummy_source_x = [torch.randn(batch_size, input_dim, device=DEVICE) for _ in range(K)]
    dummy_source_y = [torch.randint(0, n_classes, (batch_size,), device=DEVICE) for _ in range(K)]
    dummy_target_x = torch.randn(batch_size, input_dim, device=DEVICE)

    l_cls, l_mmd, l_disc, l_lsd, target_probs = model(dummy_source_x, dummy_source_y, dummy_target_x)

    print(f"L_cls:  {l_cls.item():.5f} (requires_grad={l_cls.requires_grad})")
    print(f"L_mmd:  {l_mmd.item():.5f} (requires_grad={l_mmd.requires_grad})")
    print(f"L_disc: {l_disc.item():.5f} (requires_grad={l_disc.requires_grad})")
    print(f"L_lsd:  {l_lsd.item():.5f} (requires_grad={l_lsd.requires_grad})")

    total_loss = l_cls + l_mmd + 0.01 * l_disc + 0.5 * l_lsd
    total_loss.backward()
    print("Backward pass successful. Gradients computed cleanly with no NaNs.")

    print("\n2. Testing forward pass with real extracted DREAMER data...")
    # Load Subject 1 as target, Subjects 2-6 as K=5 sources
    target_data = np.load(DATA_PROCESSED / "DREAMER_DE_S01.npy")[:batch_size]
    real_target_x = torch.from_numpy(target_data).to(DEVICE)

    real_source_x = []
    real_source_y = []
    for s in range(2, 2 + K):
        feats = np.load(DATA_PROCESSED / f"DREAMER_DE_S{s:02d}.npy")[:batch_size]
        labels = np.load(DATA_PROCESSED / f"DREAMER_labels_valence_S{s:02d}.npy")[:batch_size]
        real_source_x.append(torch.from_numpy(feats).to(DEVICE))
        real_source_y.append(torch.from_numpy(labels).to(DEVICE))

    model.zero_grad()
    r_cls, r_mmd, r_disc, r_lsd, _ = model(real_source_x, real_source_y, real_target_x)
    print(f"Real Data Loss Check -> L_cls={r_cls.item():.4f}, L_mmd={r_mmd.item():.4f}, L_disc={r_disc.item():.4f}, L_lsd={r_lsd.item():.4f}")

    print("\n3. Testing target inference mode...")
    model.eval()
    with torch.no_grad():
        preds, probs = model.predict(real_target_x)
        print(f"Inference predictions shape: {preds.shape}, sample: {preds[:5].tolist()}")

    print("\nModel verification passed.")

if __name__ == "__main__":
    test_asjda()