# src/config.py
from pathlib import Path
import torch

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_RAW = BASE_DIR / "data" / "raw" / "DREAMER.mat"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
LOG_DIR = BASE_DIR / "logs"
RESULTS_DIR = BASE_DIR / "results"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

FS = 128  # Sampling rate in Hz
WINDOW_SEC = 2  # 2-second windows yield 256 samples per window
WINDOW_LEN = FS * WINDOW_SEC

# 5 standard EEG frequency bands
BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 14),
    "beta": (14, 31),
    "gamma": (31, 50),
}