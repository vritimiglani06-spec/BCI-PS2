# src/extract_features.py
import math
import numpy as np
import scipy.io as sio
from scipy.signal import butter, sosfiltfilt
from tqdm import tqdm
from config import DATA_RAW, DATA_PROCESSED, FS, WINDOW_LEN, BANDS

def unwrap(x):
    """Recursively unpack 1x1 MATLAB object arrays down to the raw array."""
    while isinstance(x, np.ndarray) and x.dtype == object and x.size == 1:
        x = x.item()
    return x

def get_bandpass_sos(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = max(lowcut / nyq, 1e-4)
    high = min(highcut / nyq, 0.99)
    return butter(order, [low, high], btype="bandpass", output="sos")

SOS_FILTERS = {
    name: get_bandpass_sos(freq[0], freq[1], FS)
    for name, freq in BANDS.items()
}

def compute_de(segment):
    de_list = []
    for sos in SOS_FILTERS.values():
        filtered = sosfiltfilt(sos, segment, axis=0)
        var = np.var(filtered, axis=0, ddof=1)
        de = 0.5 * np.log(2.0 * math.pi * math.e * var + 1e-8)
        de_list.append(de)
    return np.concatenate(de_list)  # 14 channels x 5 bands = 70 features

def extract_all():
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    print("Loading DREAMER.mat...")
    mat = sio.loadmat(DATA_RAW)
    dreamer = mat["DREAMER"][0, 0]
    data = np.squeeze(dreamer["Data"])

    print("Extracting 2-second windowed DE features across all 23 subjects...")
    for s_idx, subj in enumerate(tqdm(data, desc="Subjects")):
        eeg = subj["EEG"][0, 0]
        
        stim_key = [k for k in eeg.dtype.names if "stim" in k.lower()][0]
        base_key = [k for k in eeg.dtype.names if "base" in k.lower()][0]
        val_key = [k for k in subj.dtype.names if "val" in k.lower()][0]
        aro_key = [k for k in subj.dtype.names if "aro" in k.lower()][0]

        stimuli = np.squeeze(unwrap(eeg[stim_key]))
        baseline = np.squeeze(unwrap(eeg[base_key]))
        val_ratings = np.squeeze(unwrap(subj[val_key])).flatten()
        aro_ratings = np.squeeze(unwrap(subj[aro_key])).flatten()

        subj_feats = []
        subj_val = []
        subj_aro = []

        for trial_idx in range(len(stimuli)):
            stim = unwrap(stimuli[trial_idx])
            base = unwrap(baseline[trial_idx])

            # Ensure (Samples, Channels) orientation
            if stim.shape[1] != 14:
                stim = stim.T
            if base.shape[1] != 14:
                base = base.T

            # 1. Mean baseline DE for this trial
            n_base_win = base.shape[0] // WINDOW_LEN
            base_des = [
                compute_de(base[w * WINDOW_LEN : (w + 1) * WINDOW_LEN, :])
                for w in range(n_base_win)
            ]
            mean_base = np.mean(base_des, axis=0) if len(base_des) > 0 else np.zeros(70)

            # 2. Stimulus windowing + baseline subtraction
            n_stim_win = stim.shape[0] // WINDOW_LEN
            y_v = 1 if val_ratings[trial_idx] > 3.0 else 0
            y_a = 1 if aro_ratings[trial_idx] > 3.0 else 0

            for w in range(n_stim_win):
                de = compute_de(stim[w * WINDOW_LEN : (w + 1) * WINDOW_LEN, :]) - mean_base
                subj_feats.append(de)
                subj_val.append(y_v)
                subj_aro.append(y_a)

        feats_arr = np.array(subj_feats, dtype=np.float32)
        val_arr = np.array(subj_val, dtype=np.int64)
        aro_arr = np.array(subj_aro, dtype=np.int64)

        assert np.all(np.isfinite(feats_arr)), f"NaN/Inf detected in Subject {s_idx+1}"

        np.save(DATA_PROCESSED / f"DREAMER_DE_S{s_idx+1:02d}.npy", feats_arr)
        np.save(DATA_PROCESSED / f"DREAMER_labels_valence_S{s_idx+1:02d}.npy", val_arr)
        np.save(DATA_PROCESSED / f"DREAMER_labels_arousal_S{s_idx+1:02d}.npy", aro_arr)

    print("Completed. All 69 .npy feature files successfully written to data/processed/")

if __name__ == "__main__":
    extract_all()