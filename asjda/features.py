from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from scipy.signal import butter, filtfilt


EEG_BANDS: Dict[str, Tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 14.0),
    "beta": (14.0, 31.0),
    "gamma": (31.0, 50.0),
}


def bandpass_filter(signal: np.ndarray, sfreq: float, low: float, high: float, order: int = 4) -> np.ndarray:
    nyq = 0.5 * sfreq
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, signal, axis=-1)


def differential_entropy(signal: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Differential entropy for Gaussian EEG segments: 0.5*log(2*pi*e*var)."""
    var = np.var(signal, axis=-1) + eps
    return 0.5 * np.log(2 * np.pi * np.e * var)


def extract_de_features(
    epochs: np.ndarray,
    sfreq: float,
    bands: Dict[str, Tuple[float, float]] = EEG_BANDS,
) -> Tuple[np.ndarray, list[str]]:
    """Extract DE features from epochs shaped [n_epochs, n_channels, n_times]."""
    if epochs.ndim != 3:
        raise ValueError("epochs must have shape [n_epochs, n_channels, n_times]")

    features = []
    names = []
    for band_name, (low, high) in bands.items():
        filtered = bandpass_filter(epochs, sfreq, low, high)
        de = differential_entropy(filtered)
        features.append(de)
        names.extend([f"de_{band_name}_ch{idx:02d}" for idx in range(epochs.shape[1])])
    return np.concatenate(features, axis=1).astype(np.float32), names


def segment_trials(trials: np.ndarray, sfreq: float, window_seconds: float = 4.0) -> np.ndarray:
    """Split DEAP-style trials [trials, channels, samples] into fixed windows."""
    if trials.ndim != 3:
        raise ValueError("trials must have shape [n_trials, n_channels, n_samples]")
    window = int(round(window_seconds * sfreq))
    if window <= 0 or window > trials.shape[-1]:
        raise ValueError("window_seconds produces an invalid segment length")

    segments = []
    for trial in trials:
        for start in range(0, trial.shape[-1] - window + 1, window):
            segments.append(trial[:, start:start + window])
    return np.stack(segments, axis=0).astype(np.float32)


def repeat_trial_labels(labels: np.ndarray, segments_per_trial: int) -> np.ndarray:
    return np.repeat(labels, segments_per_trial).astype(np.int64)
