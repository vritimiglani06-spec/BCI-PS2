# src/check_raw.py
import scipy.io as sio
import numpy as np
from config import DATA_RAW

def unwrap(x):
    """Recursively unpack 1x1 MATLAB object arrays down to the raw array."""
    while isinstance(x, np.ndarray) and x.dtype == object and x.size == 1:
        x = x.item()
    return x

def verify_dreamer():
    print(f"Checking: {DATA_RAW}")
    if not DATA_RAW.exists():
        raise FileNotFoundError(f"DREAMER.mat not found at {DATA_RAW}")

    print("Loading DREAMER.mat into memory...")
    mat = sio.loadmat(DATA_RAW)
    dreamer = mat["DREAMER"][0, 0]
    data = np.squeeze(dreamer["Data"])
    n_subjects = len(data)
    print(f"Total subjects found: {n_subjects}")
    assert n_subjects == 23, f"Expected 23 subjects, got {n_subjects}"

    s0 = data[0]
    eeg = s0["EEG"][0, 0]

    stim_key = [k for k in eeg.dtype.names if "stim" in k.lower()][0]
    base_key = [k for k in eeg.dtype.names if "base" in k.lower()][0]
    val_key = [k for k in s0.dtype.names if "val" in k.lower()][0]
    aro_key = [k for k in s0.dtype.names if "aro" in k.lower()][0]

    # Unwrap the outer 1x1 cell container
    stimuli = np.squeeze(unwrap(eeg[stim_key]))
    baseline = np.squeeze(unwrap(eeg[base_key]))
    valence = np.squeeze(unwrap(s0[val_key]))
    arousal = np.squeeze(unwrap(s0[aro_key]))

    print(f"EEG fields: Stimuli='{stim_key}', Baseline='{base_key}'")
    print(f"Trials found for Subject 1: {len(stimuli)} (expected 18)")

    t0_stim = unwrap(stimuli[0])
    t0_base = unwrap(baseline[0])
    
    # Ensure (Samples, Channels)
    if t0_stim.shape[1] != 14:
        t0_stim = t0_stim.T
    if t0_base.shape[1] != 14:
        t0_base = t0_base.T

    print(f"Subject 1 Trial 1 Stimulus EEG shape: {t0_stim.shape} (Time x Channels)")
    print(f"Subject 1 Trial 1 Baseline EEG shape: {t0_base.shape}")
    print(f"Subject 1 Valence rating shape: {valence.shape}, Arousal shape: {arousal.shape}")
    print("Verification passed.")

if __name__ == "__main__":
    verify_dreamer()