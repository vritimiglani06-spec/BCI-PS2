from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import pickle
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset


@dataclass
class SubjectData:
    subject_id: str
    x: np.ndarray
    y: np.ndarray


class ArrayDataset(Dataset):
    def __init__(self, x: np.ndarray, y: Optional[np.ndarray] = None):
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = None if y is None else torch.as_tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        if self.y is None:
            return self.x[idx]
        return self.x[idx], self.y[idx]


def load_manifest_dataset(
    manifest_csv: str | Path,
    *,
    file_col: str = "file",
    subject_col: str = "subject_id",
    label_col: str = "label",
    device_col: Optional[str] = "device",
    session_col: Optional[str] = "session",
    feature_prefixes: Optional[List[str]] = None,
    device_filter: Optional[str] = None,
    session_filter: Optional[str] = None,
) -> Tuple[Dict[str, SubjectData], LabelEncoder, List[str]]:
    """Load a subject-wise feature dataset from a manifest CSV.

    Manifest rows point to CSV files. Each CSV must contain the label column and
    numeric feature columns. Features are concatenated subject-wise.
    """
    manifest_path = Path(manifest_csv)
    manifest = pd.read_csv(manifest_path)

    if device_filter and device_col in manifest:
        manifest = manifest[manifest[device_col].astype(str) == str(device_filter)]
    if session_filter and session_col in manifest:
        manifest = manifest[manifest[session_col].astype(str) == str(session_filter)]

    rows = []
    for _, row in manifest.iterrows():
        file_path = Path(row[file_col])
        if not file_path.is_absolute():
            file_path = manifest_path.parent / file_path
        df = pd.read_csv(file_path)
        df[subject_col] = str(row[subject_col])
        rows.append(df)

    if not rows:
        raise ValueError("No data files matched the manifest filters.")

    data = pd.concat(rows, ignore_index=True)
    label_encoder = LabelEncoder()
    data[label_col] = label_encoder.fit_transform(data[label_col].astype(str))

    non_features = {subject_col, label_col}
    feature_cols = [
        c for c in data.columns
        if c not in non_features and pd.api.types.is_numeric_dtype(data[c])
    ]
    if feature_prefixes:
        feature_cols = [
            c for c in feature_cols
            if any(c.startswith(prefix) for prefix in feature_prefixes)
        ]
    if not feature_cols:
        raise ValueError("No numeric feature columns found.")

    subjects: Dict[str, SubjectData] = {}
    for subject_id, sub in data.groupby(subject_col):
        subjects[str(subject_id)] = SubjectData(
            subject_id=str(subject_id),
            x=sub[feature_cols].to_numpy(dtype=np.float32),
            y=sub[label_col].to_numpy(dtype=np.int64),
        )
    return subjects, label_encoder, feature_cols


def standardize_sources_and_target(
    sources: Iterable[SubjectData],
    target: SubjectData,
) -> Tuple[List[SubjectData], SubjectData, StandardScaler]:
    """Fit scaler on source samples only, then transform sources and target."""
    sources = list(sources)
    scaler = StandardScaler()
    scaler.fit(np.vstack([s.x for s in sources]))

    scaled_sources = [
        SubjectData(s.subject_id, scaler.transform(s.x).astype(np.float32), s.y)
        for s in sources
    ]
    scaled_target = SubjectData(
        target.subject_id,
        scaler.transform(target.x).astype(np.float32),
        target.y,
    )
    return scaled_sources, scaled_target, scaler


def make_loader(x: np.ndarray, y: Optional[np.ndarray], batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(ArrayDataset(x, y), batch_size=batch_size, shuffle=shuffle, drop_last=False)


def load_deap_dat_subject(path: str | Path, label_mode: str = "valence") -> SubjectData:
    """Load one official/preprocessed DEAP .dat subject file.

    DEAP .dat files contain:
    - data: [40 trials, 40 channels, 8064 samples]
    - labels: [40 trials, 4 ratings] for valence, arousal, dominance, liking

    The ASJDA paper uses the first 32 EEG channels and binary labels:
    rating > 5 is high, rating < 5 is low. Ratings equal to 5 are dropped.
    """
    path = Path(path)
    with path.open("rb") as f:
        payload = pickle.load(f, encoding="latin1")

    data = np.asarray(payload["data"], dtype=np.float32)[:, :32, :]
    labels = np.asarray(payload["labels"], dtype=np.float32)
    dim = {"valence": 0, "arousal": 1}[label_mode]
    scores = labels[:, dim]
    keep = scores != 5
    y = (scores[keep] > 5).astype(np.int64)
    subject_id = path.stem
    return SubjectData(subject_id=subject_id, x=data[keep], y=y)


def load_deap_dat_folder(folder: str | Path, label_mode: str = "valence") -> Dict[str, SubjectData]:
    folder = Path(folder)
    files = sorted(folder.glob("s*.dat"))
    if not files:
        files = sorted(folder.rglob("s*.dat"))
    if not files:
        raise ValueError(f"No DEAP .dat files found under {folder}")
    return {subject.subject_id: subject for subject in (load_deap_dat_subject(p, label_mode) for p in files)}
