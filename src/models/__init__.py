"""PyTorch models, datasets, and baseline benchmarks for stock price forecasting."""

from __future__ import annotations

import os
from pathlib import Path
import random
from typing import Any, Dict, Union
import numpy as np
import torch
import yaml

from src.models.dataset import StockSequenceDataset, create_dataloaders
from src.models.baseline import NaivePersistenceBaseline
from src.models.gru import StockGRU
from src.models.trainer import EarlyStopping, ModelTrainer, resolve_device


def load_config(config_path: Union[str, Path] = "configs/config.yaml") -> Dict[str, Any]:
    """Safely load and validate project YAML configuration."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    required_sections = ["data", "model", "training"]
    missing = [s for s in required_sections if s not in config]
    if missing:
        raise ValueError(f"Configuration file {path} missing required sections: {missing}")

    return config


def set_seed(seed: int = 42) -> None:
    """Set global random seed across Python, NumPy, and PyTorch for deterministic execution."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False




__all__ = [
    "load_config",
    "set_seed",
    "resolve_device",
    "StockSequenceDataset",
    "create_dataloaders",
    "NaivePersistenceBaseline",
    "StockGRU",
    "EarlyStopping",
    "ModelTrainer",
]
