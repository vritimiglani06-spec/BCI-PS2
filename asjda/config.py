from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import yaml


@dataclass
class DataConfig:
    manifest_csv: str
    subject_col: str = "subject_id"
    label_col: str = "label"
    device_col: Optional[str] = "device"
    session_col: Optional[str] = "session"
    file_col: str = "file"
    feature_prefixes: Optional[List[str]] = None
    device_filter: Optional[str] = None
    session_filter: Optional[str] = None


@dataclass
class ModelConfig:
    input_dim: Optional[int] = None
    num_classes: int = 4
    shared_hidden: int = 128
    shared_dim: int = 64
    specific_dim: int = 32
    leaky_relu_slope: float = 0.01


@dataclass
class TrainConfig:
    batch_size: int = 32
    epochs: int = 200
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    jsd_thresholds: Optional[List[float]] = None
    default_jsd_threshold: float = 0.005
    mmd_kernel_mul: float = 2.0
    mmd_kernel_num: int = 5
    seed: int = 42


@dataclass
class ExperimentConfig:
    output_dir: str = "runs/asjda"
    target_subjects: Optional[List[str]] = None
    val_fraction_for_source_threshold: float = 0.2


@dataclass
class Config:
    data: DataConfig
    model: ModelConfig
    train: TrainConfig
    experiment: ExperimentConfig


def _build(cls, values: Dict):
    fields = {f.name for f in cls.__dataclass_fields__.values()}
    return cls(**{k: v for k, v in values.items() if k in fields})


def load_config(path: str | Path) -> Config:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return Config(
        data=_build(DataConfig, raw.get("data", {})),
        model=_build(ModelConfig, raw.get("model", {})),
        train=_build(TrainConfig, raw.get("train", {})),
        experiment=_build(ExperimentConfig, raw.get("experiment", {})),
    )
